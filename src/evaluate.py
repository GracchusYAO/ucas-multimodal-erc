"""评估缓存特征上的 MELD 模型。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

from src.torch_import_patch import patch_inspect_for_torch, restore_common_builtins, stub_torch_dynamo

restore_common_builtins()
patch_inspect_for_torch()

import torch

restore_common_builtins()
stub_torch_dynamo(torch)

import yaml

restore_common_builtins()

from src.dataset import ID2EMOTION, MELD_SPLITS
from src.feature_dataset import make_dialogue_loader, make_feature_loader
from src.models import build_model


LABEL_IDS = list(range(len(ID2EMOTION)))
LABEL_NAMES = [ID2EMOTION[index] for index in LABEL_IDS]
GATED_MODELS = {
    "dgf",
    "dgf_dropout",
    "dgf_context",
    "late_fusion_hubert",
    "late_fusion_hubert_face",
    "late_fusion_hubert_stats",
    "quality_late_fusion_hubert",
    "context_residual_gated_fusion",
    "context_lstm_residual_gated_fusion",
}


def active_modalities(config: dict) -> tuple[str, ...]:
    """从 config 里读出当前模型使用哪些模态。"""
    enabled = config.get("modalities", {"text": True})
    names = (
        "text",
        "audio",
        "audio_hubert",
        "audio_hubert_stats",
        "audio_prosody",
        "audio_hubert_prosody",
        "audio_emotion",
        "visual",
        "visual_face",
        "visual_expression",
        "visual_expression_affectnet",
        "visual_expression_topk",
        "visual_expression_compact",
        "visual_clip_expression",
    )
    return tuple(name for name in names if enabled.get(name, False))


def build_quality_tensor(batch: dict, modalities: tuple[str, ...], device: torch.device) -> torch.Tensor:
    """把各模态 available 标记整理成 [B, M]。"""
    batch_size = batch["label"].size(0)
    quality = torch.ones(batch_size, len(modalities), device=device)
    for index, modality in enumerate(modalities):
        key = f"{modality}_available"
        if key in batch:
            quality[:, index] = batch[key].to(device).float()
    return quality


def choose_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


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
def collect_predictions(
    model: torch.nn.Module,
    loader,
    modalities: tuple[str, ...],
    use_context: bool,
    device: torch.device,
    collect_gates: bool = False,
    zero_modalities: tuple[str, ...] = (),
) -> tuple[list[int], list[int], list[str], list[list[float]]]:
    model.eval()
    predictions: list[int] = []
    gold_labels: list[int] = []
    keys: list[str] = []
    gate_weights: list[list[float]] = []

    for batch in loader:
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device) if use_context else None
        inputs = []
        for modality in modalities:
            feature = batch[modality].to(device)
            if modality in zero_modalities:
                feature = torch.zeros_like(feature)  # 评估缺失模态时，直接把该模态置零
            inputs.append(feature)

        quality = None
        if getattr(model, "uses_quality", False):
            quality = build_quality_tensor(batch, modalities, device)
            for modality in zero_modalities:
                if modality in modalities:
                    quality[:, modalities.index(modality)] = 0.0  # 被遮掉的模态也标成不可用

        if collect_gates:
            if quality is not None:
                logits, gates = model(*inputs, quality=quality, return_gate=True)
            elif use_context:
                text, audio, visual = inputs
                logits, gates = model(text, audio, visual, mask=mask, return_gate=True)
            else:
                text, audio, visual = inputs
                logits, gates = model(text, audio, visual, return_gate=True)
        else:
            if quality is not None:
                logits = model(*inputs, quality=quality)
            elif use_context:
                logits = model(*inputs, mask=mask)
            else:
                logits = model(*inputs)
            gates = None

        if mask is None:
            valid_logits = logits
            valid_labels = labels
            valid_gates = gates
        else:
            valid_logits = logits[mask]  # 只评估真实 utterance，不评估 padding
            valid_labels = labels[mask]
            valid_gates = gates[mask] if gates is not None else None

        predictions.extend(valid_logits.argmax(dim=1).cpu().tolist())
        gold_labels.extend(valid_labels.cpu().tolist())
        keys.extend(flatten_keys(batch, use_context))
        if valid_gates is not None:
            gate_weights.extend(valid_gates.cpu().tolist())

    return gold_labels, predictions, keys, gate_weights


def build_metrics(gold_labels: list[int], predictions: list[int]) -> dict:
    matrix = confusion_matrix_counts(gold_labels, predictions).float()
    total = matrix.sum().clamp(min=1.0)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = true_positive / predicted.clamp(min=1.0)
    recall = true_positive / support.clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return {
        "accuracy": float(true_positive.sum().item() / total.item()),
        "weighted_f1": float((f1 * support).sum().item() / total.item()),
        "macro_f1": float(f1.mean().item()),
        "per_class": {
            name: {
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
                "support": int(support[index].item()),
            }
            for index, name in enumerate(LABEL_NAMES)
        },
    }


def save_predictions(
    path: Path,
    keys: list[str],
    gold_labels: list[int],
    predictions: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "gold_id", "gold", "pred_id", "pred", "correct"])
        for key, gold, pred in zip(keys, gold_labels, predictions):
            writer.writerow([key, gold, ID2EMOTION[gold], pred, ID2EMOTION[pred], int(gold == pred)])


def save_gate_weights(
    path: Path,
    keys: list[str],
    gold_labels: list[int],
    predictions: list[int],
    gate_weights: list[list[float]],
) -> None:
    """保存每条 utterance 的 text/audio/visual gate，后面画图和分析会用。"""
    if not gate_weights:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "key",
                "gold_id",
                "gold",
                "pred_id",
                "pred",
                "correct",
                "gate_text",
                "gate_audio",
                "gate_visual",
            ]
        )
        for key, gold, pred, gates in zip(keys, gold_labels, predictions, gate_weights):
            writer.writerow(
                [
                    key,
                    gold,
                    ID2EMOTION[gold],
                    pred,
                    ID2EMOTION[pred],
                    int(gold == pred),
                    gates[0],
                    gates[1],
                    gates[2],
                ]
            )


def save_confusion_matrix(output_dir: Path, gold_labels: list[int], predictions: list[int]) -> None:
    matrix = confusion_matrix_counts(gold_labels, predictions)

    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["gold\\pred", *LABEL_NAMES])
        for name, row in zip(LABEL_NAMES, matrix):
            writer.writerow([name, *row.tolist()])

    # 当前 WSL/conda 环境里 matplotlib 偶发污染全局状态；评估阶段只保存 CSV。


def confusion_matrix_counts(gold_labels: list[int], predictions: list[int]) -> torch.Tensor:
    """纯 PyTorch 混淆矩阵，避免评估脚本依赖 sklearn/scipy。"""
    labels = torch.tensor(gold_labels, dtype=torch.long)
    preds = torch.tensor(predictions, dtype=torch.long)
    num_classes = len(LABEL_IDS)
    flat_index = labels * num_classes + preds
    return torch.bincount(flat_index, minlength=num_classes * num_classes).view(num_classes, num_classes)


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = config.get("model_name", checkpoint.get("model_name", "model"))
    use_context = bool(config.get("use_context", False))
    modalities = active_modalities(config)
    collect_gates = (
        not args.no_save_gates
        and model_name in GATED_MODELS
        and len(modalities) == 3
    )
    zero_modalities = tuple(args.zero_modality or [])

    seed = int(config.get("seed", 114514))
    set_seed(seed)
    device = choose_device(args.device)

    batch_size = args.batch_size or int(
        config.get("batch_size_utterance", config.get("batch_size_dialogue", 64))
    )
    loader_fn = make_dialogue_loader if use_context else make_feature_loader
    loader = loader_fn(
        args.split,
        modalities=modalities,
        features_root=args.features_root,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    gold_labels, predictions, keys, gate_weights = collect_predictions(
        model,
        loader,
        modalities,
        use_context,
        device,
        collect_gates=collect_gates,
        zero_modalities=zero_modalities,
    )
    metrics = build_metrics(gold_labels, predictions)

    output_dir = Path(args.output_dir or f"results/evaluate/{model_name}_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_with_meta = {
        "model_name": model_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "zero_modalities": zero_modalities,
        "dev_metrics": checkpoint.get("dev_metrics"),
        **metrics,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_with_meta, file, indent=2)
    save_predictions(output_dir / "predictions.csv", keys, gold_labels, predictions)
    save_confusion_matrix(output_dir, gold_labels, predictions)
    save_gate_weights(output_dir / "gate_weights.csv", keys, gold_labels, predictions, gate_weights)

    print(
        f"{model_name} {args.split}: "
        f"accuracy={metrics['accuracy']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    if zero_modalities:
        print(f"zero_modalities={zero_modalities}")
    print(f"saved {output_dir}")
    if gate_weights:
        print(f"saved gate weights: {output_dir / 'gate_weights.csv'}")
    return metrics_with_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MELD model checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=MELD_SPLITS)
    parser.add_argument("--features-root", default="features")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-save-gates", action="store_true")
    parser.add_argument(
        "--zero-modality",
        action="append",
        choices=(
            "text",
            "audio",
            "audio_hubert",
            "audio_hubert_stats",
            "audio_prosody",
            "audio_hubert_prosody",
            "audio_emotion",
            "visual",
            "visual_face",
            "visual_expression",
            "visual_expression_affectnet",
            "visual_expression_topk",
            "visual_expression_compact",
            "visual_clip_expression",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
