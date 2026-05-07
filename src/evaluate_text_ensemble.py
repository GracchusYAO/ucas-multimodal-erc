"""对多个文本微调模型做 logits ensemble。

用法示例：
python -m src.evaluate_text_ensemble \
  --config configs/text_finetune.yaml --checkpoint checkpoints/text_finetune/best_text_finetune.pt \
  --config configs/text_finetune_context_unweighted.yaml --checkpoint checkpoints/text_finetune_context_unweighted/best_text_finetune_context_unweighted.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import ID2EMOTION, MELD_SPLITS
from src.train_text_finetune import (
    TextUtteranceDataset,
    TransformerERCClassifier,
    build_collate_fn,
    choose_device,
    load_tokenizer,
    set_seed,
)


LABEL_IDS = list(ID2EMOTION)
LABEL_NAMES = [ID2EMOTION[index] for index in LABEL_IDS]


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def make_loader(config: dict, split: str, data_root: str | Path, batch_size: int, num_workers: int):
    tokenizer = load_tokenizer(config.get("text_model", "FacebookAI/roberta-base"), config)
    dataset = TextUtteranceDataset(
        split,
        data_root=data_root,
        context_window=int(config.get("context_window", 0)),
        include_speaker=bool(config.get("include_speaker", True)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=build_collate_fn(tokenizer, int(config.get("max_length", 128))),
    )


@torch.no_grad()
def collect_logits(
    config: dict,
    checkpoint_path: str | Path,
    split: str,
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """单个 checkpoint 输出完整 split 的 logits，后面直接平均。"""
    loader = make_loader(config, split, data_root, batch_size, num_workers)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TransformerERCClassifier(
        config.get("text_model", "FacebookAI/roberta-base"),
        num_classes=int(config.get("num_classes", 7)),
        dropout=float(config.get("dropout", 0.0)),
        pooling=str(config.get("pooling", "cls")),
        local_files_only=bool(config.get("local_files_only", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_logits = []
    all_labels = []
    all_keys = []
    for batch in loader:
        encoded = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        logits = model(**encoded)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].long())
        all_keys.extend(batch["key"])

    return torch.cat(all_logits), torch.cat(all_labels), all_keys


def build_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict:
    gold = labels.tolist()
    pred = predictions.tolist()
    per_class = {}
    weighted_f1 = 0.0
    macro_f1 = 0.0
    total = len(gold)
    correct = sum(1 for gold_item, pred_item in zip(gold, pred) if gold_item == pred_item)
    for label_id, label_name in zip(LABEL_IDS, LABEL_NAMES):
        tp = sum(1 for gold_item, pred_item in zip(gold, pred) if gold_item == label_id and pred_item == label_id)
        fp = sum(1 for gold_item, pred_item in zip(gold, pred) if gold_item != label_id and pred_item == label_id)
        fn = sum(1 for gold_item, pred_item in zip(gold, pred) if gold_item == label_id and pred_item != label_id)
        support = sum(1 for gold_item in gold if gold_item == label_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        weighted_f1 += f1 * support
        macro_f1 += f1

    return {
        "accuracy": correct / total if total else 0.0,
        "weighted_f1": weighted_f1 / total if total else 0.0,
        "macro_f1": macro_f1 / len(LABEL_IDS),
        "per_class": per_class,
    }


def average_logits(logits_list: list[torch.Tensor], average: str, weights: torch.Tensor | None = None) -> torch.Tensor:
    stacked = torch.stack(logits_list)
    if weights is None:
        weights = torch.full((stacked.size(0),), 1.0 / stacked.size(0))
    weights = weights.to(stacked).view(-1, 1, 1)
    if average == "prob":
        return (torch.softmax(stacked, dim=2) * weights).sum(dim=0)
    return (stacked * weights).sum(dim=0)


def search_ensemble_weights(
    logits_list: list[torch.Tensor],
    labels: torch.Tensor,
    average: str,
    trials: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    """在 dev 上随机搜索 ensemble 权重，用 weighted F1 选最优。"""
    generator = torch.Generator().manual_seed(seed)
    model_count = len(logits_list)
    candidates = [torch.full((model_count,), 1.0 / model_count)]
    candidates.extend(torch.eye(model_count))
    if model_count >= 2:
        random_weights = torch.rand(trials, model_count, generator=generator)
        random_weights = random_weights / random_weights.sum(dim=1, keepdim=True)
        candidates.extend(random_weights)

    best_weights = candidates[0]
    best_metrics = build_metrics(labels, average_logits(logits_list, average, best_weights).argmax(dim=1))
    for weights in candidates[1:]:
        predictions = average_logits(logits_list, average, weights).argmax(dim=1)
        metrics = build_metrics(labels, predictions)
        if metrics["weighted_f1"] > best_metrics["weighted_f1"]:
            best_weights = weights
            best_metrics = metrics
    return best_weights, best_metrics


def save_predictions(path: Path, keys: list[str], labels: torch.Tensor, predictions: torch.Tensor) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "gold_id", "gold", "pred_id", "pred", "correct"])
        for key, gold, pred in zip(keys, labels.tolist(), predictions.tolist()):
            writer.writerow([key, gold, ID2EMOTION[gold], pred, ID2EMOTION[pred], int(gold == pred)])


def evaluate_ensemble(args: argparse.Namespace) -> dict:
    if len(args.config) != len(args.checkpoint):
        raise ValueError("--config and --checkpoint must have the same count.")

    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [load_config(path) for path in args.config]
    run_items = [
        {"config": config_path, "checkpoint": checkpoint_path}
        for config_path, checkpoint_path in zip(args.config, args.checkpoint)
    ]

    weights = None
    tune_metrics = None
    if args.tune_weights_on_dev:
        dev_logits_list, dev_labels, _ = collect_all_logits(
            configs,
            args.checkpoint,
            "dev",
            args.data_root,
            args.batch_size,
            args.num_workers,
            device,
        )
        weights, tune_metrics = search_ensemble_weights(
            dev_logits_list,
            dev_labels,
            args.average,
            args.weight_search_trials,
            args.seed,
        )
        print(
            "tuned dev: "
            f"weighted_f1={tune_metrics['weighted_f1']:.4f} "
            f"macro_f1={tune_metrics['macro_f1']:.4f} "
            f"weights={[round(item, 4) for item in weights.tolist()]}"
        )

    logits_list, reference_labels, reference_keys = collect_all_logits(
        configs,
        args.checkpoint,
        args.split,
        args.data_root,
        args.batch_size,
        args.num_workers,
        device,
    )

    scores = average_logits(logits_list, args.average, weights)
    predictions = scores.argmax(dim=1)
    metrics = build_metrics(reference_labels, predictions)
    metrics_with_meta = {
        "split": args.split,
        "average": args.average,
        "runs": run_items,
        "ensemble_weights": weights.tolist() if weights is not None else None,
        "tune_dev_metrics": tune_metrics,
        **metrics,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_with_meta, file, indent=2)
    save_predictions(output_dir / "predictions.csv", reference_keys, reference_labels, predictions)

    print(
        f"ensemble {args.split}: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"saved {output_dir}")
    return metrics_with_meta


def collect_all_logits(
    configs: list[dict],
    checkpoints: list[str],
    split: str,
    data_root: str | Path,
    batch_size_override: int | None,
    num_workers: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, list[str]]:
    logits_list = []
    reference_labels = None
    reference_keys = None

    for config, checkpoint_path in zip(configs, checkpoints):
        batch_size = batch_size_override or int(config.get("batch_size", 8))
        logits, labels, keys = collect_logits(
            config,
            checkpoint_path,
            split,
            data_root,
            batch_size,
            num_workers,
            device,
        )
        if reference_labels is None:
            reference_labels = labels
            reference_keys = keys
        elif not torch.equal(reference_labels, labels) or reference_keys != keys:
            raise ValueError(f"Prediction order mismatch for {checkpoint_path}")

        logits_list.append(logits)
        print(f"loaded logits: {checkpoint_path} {tuple(logits.shape)}")

    return logits_list, reference_labels, reference_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of fine-tuned text models.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--split", default="test", choices=MELD_SPLITS)
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default="results/text_finetune_ensemble")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--average", choices=("logit", "prob"), default="prob")
    parser.add_argument("--tune-weights-on-dev", action="store_true")
    parser.add_argument("--weight-search-trials", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_ensemble(args)


if __name__ == "__main__":
    main()
