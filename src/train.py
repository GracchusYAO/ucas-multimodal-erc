"""训练缓存特征上的 MELD 模型。

当前支持 text-only、Text+Audio+Visual concat baseline 和 DGF。
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.feature_dataset import make_dialogue_loader, make_feature_loader
from src.models import build_model


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(device: str | None) -> torch.device:
    if device:  # 命令行显式指定时优先使用
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 固定输入下避免 cudnn 自动换算法
    torch.backends.cudnn.deterministic = True  # 尽量让 GPU 训练可复现


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """按训练集类别频次生成 weighted cross entropy 的权重。"""
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    return counts.sum() / (counts.clamp(min=1) * num_classes)  # 少数类权重大一些


def active_modalities(config: dict) -> tuple[str, ...]:
    """从 config 里读出当前模型使用哪些模态。"""
    enabled = config.get("modalities", {"text": True})
    return tuple(name for name in ("text", "audio", "visual") if enabled.get(name, False))


def forward_batch(
    model: nn.Module,
    batch: dict,
    modalities: tuple[str, ...],
    device: torch.device,
    use_context: bool,
) -> torch.Tensor:
    """把 batch 中需要的模态取出来，按顺序喂给模型。"""
    inputs = [batch[modality].to(device) for modality in modalities]  # text/audio/visual
    if use_context:
        return model(*inputs, mask=batch["mask"].to(device))  # dialogue batch 需要 mask
    return model(*inputs)


def batch_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """普通 batch 和 dialogue padding batch 共用的 loss 计算。"""
    if mask is None:
        return criterion(logits, labels), labels.size(0)

    valid_logits = logits[mask]  # 只取真实 utterance，跳过 padding
    valid_labels = labels[mask]
    return criterion(valid_logits, valid_labels), valid_labels.size(0)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modalities: tuple[str, ...],
    use_context: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        labels = batch["label"].to(device)  # 普通: [B]；context: [B, L]
        mask = batch["mask"].to(device) if use_context else None

        optimizer.zero_grad()
        logits = forward_batch(model, batch, modalities, device, use_context)
        loss, item_count = batch_loss(logits, labels, criterion, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * item_count
        total_count += item_count

    return total_loss / total_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    modalities: tuple[str, ...],
    use_context: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions: list[int] = []
    gold_labels: list[int] = []

    for batch in loader:
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device) if use_context else None
        logits = forward_batch(model, batch, modalities, device, use_context)
        loss, item_count = batch_loss(logits, labels, criterion, mask)

        total_loss += loss.item() * item_count
        total_count += item_count

        if mask is None:
            valid_logits = logits
            valid_labels = labels
        else:
            valid_logits = logits[mask]
            valid_labels = labels[mask]
        predictions.extend(valid_logits.argmax(dim=1).cpu().tolist())  # logits 最大的位置就是预测类别
        gold_labels.extend(valid_labels.cpu().tolist())

    return {
        "loss": total_loss / total_count,
        "accuracy": accuracy_score(gold_labels, predictions),
        "weighted_f1": f1_score(gold_labels, predictions, average="weighted", zero_division=0),
        "macro_f1": f1_score(gold_labels, predictions, average="macro", zero_division=0),
    }


def save_epoch_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "dev_loss",
        "dev_accuracy",
        "dev_weighted_f1",
        "dev_macro_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model_name: str,
    modalities: tuple[str, ...],
    use_context: bool,
    config: dict,
    seed: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dev_metrics: dict[str, float],
) -> None:
    """保存训练状态；best 用来交作业，last 用来断点复盘。"""
    torch.save(
        {
            "model_name": model_name,
            "modalities": modalities,
            "use_context": use_context,
            "config": config,
            "seed": seed,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dev_metrics": dev_metrics,
        },
        path,
    )


def train_model(args: argparse.Namespace, config: dict) -> None:
    model_name = config.get("model_name", "text_only")
    modalities = active_modalities(config)
    use_context = bool(config.get("use_context", False))
    if not modalities:
        raise ValueError("At least one modality must be enabled in config.")

    seed = args.seed if args.seed is not None else int(config.get("seed", 114514))
    set_seed(seed)
    device = choose_device(args.device)

    batch_size = args.batch_size or int(
        config.get("batch_size_utterance", config.get("batch_size_dialogue", 64))
    )
    max_epochs = args.max_epochs or int(config.get("max_epochs", 50))
    patience = args.patience or int(config.get("early_stopping_patience", 5))
    learning_rate = args.learning_rate or float(config.get("learning_rate", 1e-4))
    weight_decay = args.weight_decay or float(config.get("weight_decay", 1e-4))
    num_classes = int(config.get("num_classes", 7))

    loader_fn = make_dialogue_loader if use_context else make_feature_loader
    train_loader = loader_fn(
        "train",
        modalities=modalities,
        features_root=args.features_root,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dev_loader = loader_fn(
        "dev",
        modalities=modalities,
        features_root=args.features_root,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(config).to(device)
    weights = class_weights(train_loader.dataset.labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    output_dir = Path(args.output_dir or f"results/{model_name}")
    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/{model_name}")
    log_path = output_dir / "train_log.csv"
    best_path = checkpoint_dir / f"best_{model_name}.pt"
    last_path = checkpoint_dir / f"last_{model_name}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={model_name} modalities={modalities} context={use_context} seed={seed}")
    print(f"device={device} train={len(train_loader.dataset)} dev={len(dev_loader.dataset)}")
    print(f"batch_size={batch_size} lr={learning_rate} weight_decay={weight_decay}")
    print(f"log_path={log_path}")
    print(f"best_checkpoint={best_path}")
    print(f"last_checkpoint={last_path}")

    rows: list[dict[str, float | int]] = []
    best_weighted_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            modalities,
            use_context,
        )
        dev_metrics = evaluate(model, dev_loader, criterion, device, modalities, use_context)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        rows.append(row)
        save_epoch_log(log_path, rows)  # 每个 epoch 都写一次，训练中断也能看到已有结果
        save_checkpoint(
            last_path,
            model_name,
            modalities,
            use_context,
            config,
            seed,
            epoch,
            model,
            optimizer,
            dev_metrics,
        )

        improved = dev_metrics["weighted_f1"] > best_weighted_f1
        if improved:
            best_weighted_f1 = dev_metrics["weighted_f1"]
            bad_epochs = 0
            save_checkpoint(
                best_path,
                model_name,
                modalities,
                use_context,
                config,
                seed,
                epoch,
                model,
                optimizer,
                dev_metrics,
            )
        else:
            bad_epochs += 1

        marker = "*" if improved else " "
        print(
            f"{marker} epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} "
            f"dev_weighted_f1={dev_metrics['weighted_f1']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

        if bad_epochs >= patience:
            print(f"early stopping at epoch {epoch}; best_dev_weighted_f1={best_weighted_f1:.4f}")
            break

    print(f"done. best_dev_weighted_f1={best_weighted_f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MELD models from cached features.")
    parser.add_argument("--config", default="configs/text_only.yaml")
    parser.add_argument("--features-root", default="features")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_model(args, config)


if __name__ == "__main__":
    main()
