"""混合 ensemble：fine-tuned text logits + cached-feature multimodal logits。

这个脚本用于回答一个很实际的问题：
强文本模型已经很强了，冻结特征多模态模型是否还能提供互补信息？
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.torch_import_patch import patch_inspect_for_torch, restore_common_builtins, stub_torch_dynamo

restore_common_builtins()
patch_inspect_for_torch()

import torch

restore_common_builtins()
stub_torch_dynamo(torch)

from src.evaluate_text_ensemble import (
    average_logits,
    build_metrics,
    collect_logits as collect_text_logits,
    load_config,
    save_predictions,
    search_ensemble_weights,
)
from src.feature_dataset import make_dialogue_loader, make_feature_loader
from src.models import build_model
from src.train import active_modalities, build_quality_tensor, choose_device, set_seed


def flatten_keys(batch: dict, use_context: bool) -> list[str]:
    """把普通 batch 或 dialogue batch 的 key 拉平成 utterance 级列表。"""
    if not use_context:
        return list(batch["key"])

    keys = []
    mask = batch["mask"]
    for row, dialogue_keys in enumerate(batch["key"]):
        length = int(mask[row].sum().item())
        keys.extend(dialogue_keys[:length])
    return keys


@torch.no_grad()
def collect_feature_logits(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    features_root: str | Path,
    batch_size_override: int | None,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """读取 cached-feature 模型的 logits，顺序保持和 MELD split 一致。"""
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    modalities = active_modalities(config)
    use_context = bool(config.get("use_context", False))
    batch_size = batch_size_override or int(
        config.get("batch_size_utterance", config.get("batch_size_dialogue", 64))
    )

    loader_fn = make_dialogue_loader if use_context else make_feature_loader
    loader = loader_fn(
        split,
        modalities=modalities,
        features_root=features_root,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits_list = []
    labels_list = []
    keys: list[str] = []
    for batch in loader:
        inputs = [batch[modality].to(device) for modality in modalities]
        if getattr(model, "uses_quality", False):
            logits = model(*inputs, quality=build_quality_tensor(batch, modalities, device))
        elif use_context:
            logits = model(*inputs, mask=batch["mask"].to(device))
        else:
            logits = model(*inputs)

        if use_context:
            mask = batch["mask"].to(device)
            logits_list.append(logits[mask].cpu())
            labels_list.append(batch["label"][mask.cpu()].long())
        else:
            logits_list.append(logits.cpu())
            labels_list.append(batch["label"].long())
        keys.extend(flatten_keys(batch, use_context))

    return torch.cat(logits_list), torch.cat(labels_list), keys


def collect_all_logits(
    args: argparse.Namespace,
    split: str,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, list[str], list[str]]:
    logits_list: list[torch.Tensor] = []
    reference_labels = None
    reference_keys = None
    names: list[str] = []

    for config_path, checkpoint_path in zip(args.text_config or [], args.text_checkpoint or []):
        config = load_config(config_path)
        logits, labels, keys = collect_text_logits(
            config,
            checkpoint_path,
            split,
            args.data_root,
            args.batch_size,
            args.num_workers,
            device,
        )
        names.append(Path(checkpoint_path).parent.name)
        logits_list, reference_labels, reference_keys = append_checked(
            logits_list, reference_labels, reference_keys, logits, labels, keys, checkpoint_path
        )
        print(f"loaded text logits: {checkpoint_path} {tuple(logits.shape)}")

    for config_path, checkpoint_path in zip(args.feature_config or [], args.feature_checkpoint or []):
        logits, labels, keys = collect_feature_logits(
            config_path,
            checkpoint_path,
            split,
            args.features_root,
            args.batch_size,
            args.num_workers,
            device,
        )
        names.append(Path(checkpoint_path).parent.name)
        logits_list, reference_labels, reference_keys = append_checked(
            logits_list, reference_labels, reference_keys, logits, labels, keys, checkpoint_path
        )
        print(f"loaded feature logits: {checkpoint_path} {tuple(logits.shape)}")

    if not logits_list or reference_labels is None or reference_keys is None:
        raise ValueError("At least one text or feature checkpoint is required.")
    return logits_list, reference_labels, reference_keys, names


def append_checked(
    logits_list: list[torch.Tensor],
    reference_labels: torch.Tensor | None,
    reference_keys: list[str] | None,
    logits: torch.Tensor,
    labels: torch.Tensor,
    keys: list[str],
    source: str | Path,
) -> tuple[list[torch.Tensor], torch.Tensor, list[str]]:
    """确认不同模型输出顺序一致，再加入 ensemble。"""
    if reference_labels is None or reference_keys is None:
        reference_labels = labels
        reference_keys = keys
    elif not torch.equal(reference_labels, labels) or reference_keys != keys:
        raise ValueError(f"Prediction order mismatch for {source}")
    logits_list.append(logits)
    return logits_list, reference_labels, reference_keys


def evaluate_mixed_ensemble(args: argparse.Namespace) -> dict:
    if len(args.text_config or []) != len(args.text_checkpoint or []):
        raise ValueError("--text-config and --text-checkpoint must have the same count.")
    if len(args.feature_config or []) != len(args.feature_checkpoint or []):
        raise ValueError("--feature-config and --feature-checkpoint must have the same count.")

    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = None
    tune_metrics = None
    if args.tune_weights_on_dev:
        dev_logits, dev_labels, _, names = collect_all_logits(args, "dev", device)
        weights, tune_metrics = search_ensemble_weights(
            dev_logits,
            dev_labels,
            args.average,
            args.weight_search_trials,
            args.seed,
        )
        print(
            "tuned dev: "
            f"weighted_f1={tune_metrics['weighted_f1']:.4f} "
            f"macro_f1={tune_metrics['macro_f1']:.4f} "
            f"weights={format_weights(names, weights)}"
        )

    logits_list, labels, keys, names = collect_all_logits(args, args.split, device)
    scores = average_logits(logits_list, args.average, weights)
    predictions = scores.argmax(dim=1)
    metrics = build_metrics(labels, predictions)
    save_predictions(output_dir / "predictions.csv", keys, labels, predictions)

    metrics_with_meta = {
        "split": args.split,
        "average": args.average,
        "model_names": names,
        "ensemble_weights": weights.tolist() if weights is not None else None,
        "tune_dev_metrics": tune_metrics,
        **metrics,
    }
    import json

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_with_meta, file, indent=2)

    print(
        f"mixed ensemble {args.split}: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"saved {output_dir}")
    return metrics_with_meta


def format_weights(names: list[str], weights: torch.Tensor) -> list[str]:
    return [f"{name}:{weight:.4f}" for name, weight in zip(names, weights.tolist())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mixed text + multimodal ensemble.")
    parser.add_argument("--text-config", action="append")
    parser.add_argument("--text-checkpoint", action="append")
    parser.add_argument("--feature-config", action="append")
    parser.add_argument("--feature-checkpoint", action="append")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--features-root", default="features")
    parser.add_argument("--output-dir", default="results/mixed_ensemble")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--average", choices=("logit", "prob"), default="prob")
    parser.add_argument("--tune-weights-on-dev", action="store_true")
    parser.add_argument("--weight-search-trials", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    evaluate_mixed_ensemble(parse_args())


if __name__ == "__main__":
    main()
