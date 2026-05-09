"""微调 Transformer 文本编码器。

前面的 text-only baseline 是“冻结 RoBERTa 后抽特征”，这里改成直接微调预训练文本模型。
这是当前最值得先试的增强方向，因为缺失模态分析显示 audio/visual 特征暂时没有帮上忙。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # 当前环境里 torch._dynamo 偶发导入异常
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.torch_import_patch import patch_inspect_for_torch, restore_common_builtins, stub_torch_dynamo

restore_common_builtins()
patch_inspect_for_torch()

import torch

restore_common_builtins()

import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

restore_common_builtins()
stub_torch_dynamo(torch)

from src.dataset import ID2EMOTION, MELDUtterance, load_meld_split


class TextUtteranceDataset(Dataset):
    """只读取 utterance 文本和标签，给文本 encoder 微调用。"""

    def __init__(
        self,
        split: str,
        data_root: str | Path = "data/meld",
        max_samples: int | None = None,
        context_window: int = 0,
        include_speaker: bool = True,
    ) -> None:
        self.items = load_meld_split(split, data_root)
        self.texts = build_context_texts(self.items, context_window, include_speaker)
        if max_samples is not None:
            self.items = self.items[:max_samples]  # smoke test 时快速截一小段
            self.texts = self.texts[:max_samples]
        self.labels = torch.tensor([item.emotion_id for item in self.items], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return {"item": self.items[index], "text": self.texts[index]}


def build_context_texts(
    utterances: list[MELDUtterance],
    context_window: int,
    include_speaker: bool,
) -> list[str]:
    """给每条样本拼接前 context_window 句上文；0 就是原始单句。"""
    if context_window <= 0:
        if include_speaker:
            return [f"{item.speaker}: {item.text}" for item in utterances]
        return [item.text for item in utterances]

    grouped: dict[int, list[MELDUtterance]] = {}
    for item in utterances:
        grouped.setdefault(item.dialogue_id, []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: item.utterance_id)

    text_by_key = {}
    for items in grouped.values():
        for index, item in enumerate(items):
            start = max(0, index - context_window)
            pieces = []
            for ctx_item in items[start : index + 1]:
                role = "Target" if ctx_item.key == item.key else "Context"
                speaker = f"{ctx_item.speaker}: " if include_speaker else ""
                pieces.append(f"{role}: {speaker}{ctx_item.text}")
            text_by_key[item.key] = " </s> ".join(pieces)  # 用分隔符隔开上下文和目标句

    return [text_by_key[item.key] for item in utterances]


class TransformerERCClassifier(nn.Module):
    """Transformer encoder + 简单分类头，可用于 RoBERTa / DeBERTa。"""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 7,
        dropout: float = 0.2,
        pooling: str = "cls",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = load_encoder(model_name, local_files_only)
        self.pooling = pooling
        hidden_size = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # 小数据集微调时防止过拟合
            nn.Linear(hidden_size, num_classes),
        )

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]  # 第一个 token 作为整句表示
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return self.classifier(pooled)


def load_encoder(model_name: str, local_files_only: bool):
    """加载 Transformer encoder；RoBERTa 可以关 pooler，DeBERTa 不支持这个参数。"""
    try:
        return AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            add_pooling_layer=False,  # 我们自己做 cls/mean pooling，不需要 RoBERTa pooler
        )
    except TypeError:
        return AutoModel.from_pretrained(model_name, local_files_only=local_files_only)


def load_tokenizer(model_name: str, config: dict):
    """DeBERTa-v3 的 slow tokenizer 对 byte fallback 更忠实；RoBERTa 继续走默认 fast tokenizer。"""
    kwargs = {"local_files_only": bool(config.get("local_files_only", False))}
    if "tokenizer_use_fast" in config:
        kwargs["use_fast"] = bool(config["tokenizer_use_fast"])
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    return counts.sum() / (counts.clamp(min=1) * num_classes)  # 类别越少，loss 权重越高


def compute_basic_metrics(gold_labels: list[int], predictions: list[int]) -> dict[str, float]:
    """不用 sklearn，手写 accuracy / weighted F1 / macro F1，减少导入依赖。"""
    total = len(gold_labels)
    correct = sum(int(gold == pred) for gold, pred in zip(gold_labels, predictions))
    weighted_f1 = 0.0
    macro_f1 = 0.0

    for label_id in ID2EMOTION:
        tp = sum(1 for gold, pred in zip(gold_labels, predictions) if gold == label_id and pred == label_id)
        fp = sum(1 for gold, pred in zip(gold_labels, predictions) if gold != label_id and pred == label_id)
        fn = sum(1 for gold, pred in zip(gold_labels, predictions) if gold == label_id and pred != label_id)
        support = sum(1 for gold in gold_labels if gold == label_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weighted_f1 += f1 * support
        macro_f1 += f1

    return {
        "accuracy": correct / total if total else 0.0,
        "weighted_f1": weighted_f1 / total if total else 0.0,
        "macro_f1": macro_f1 / len(ID2EMOTION),
    }


def build_collate_fn(tokenizer, max_length: int):
    def collate(items: list[dict]) -> dict:
        encoded = tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        utterances = [item["item"] for item in items]
        encoded["label"] = torch.tensor([item.emotion_id for item in utterances], dtype=torch.long)
        encoded["key"] = [item.key for item in utterances]
        return encoded

    return collate


def make_loader(
    split: str,
    tokenizer,
    data_root: str | Path,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: int | None = None,
    context_window: int = 0,
    include_speaker: bool = True,
) -> DataLoader:
    dataset = TextUtteranceDataset(
        split,
        data_root,
        max_samples=max_samples,
        context_window=context_window,
        include_speaker=include_speaker,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=build_collate_fn(tokenizer, max_length),
    )


def move_batch(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    labels = batch["label"].to(device)
    encoded = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    return encoded, labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions: list[int] = []
    gold_labels: list[int] = []
    rows: list[dict[str, object]] = []

    for batch in loader:
        encoded, labels = move_batch(batch, device)
        logits = model(**encoded)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        predictions.extend(preds.cpu().tolist())
        gold_labels.extend(labels.cpu().tolist())
        for key, gold, pred in zip(batch["key"], labels.cpu().tolist(), preds.cpu().tolist()):
            rows.append(
                {
                    "key": key,
                    "gold_id": gold,
                    "gold": ID2EMOTION[gold],
                    "pred_id": pred,
                    "pred": ID2EMOTION[pred],
                    "correct": int(gold == pred),
                }
            )

    metrics = {"loss": total_loss / max(total_count, 1), **compute_basic_metrics(gold_labels, predictions)}
    return metrics, rows


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        encoded, labels = move_batch(batch, device)
        amp_context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with amp_context:
            logits = model(**encoded)
            loss = criterion(logits, labels)
            scaled_loss = loss / grad_accum_steps  # 累积梯度时 loss 要除一下

        scaler.scale(scaled_loss).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)

    return total_loss / max(total_count, 1)


def save_epoch_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "dev_loss",
                "dev_accuracy",
                "dev_weighted_f1",
                "dev_macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["key", "gold_id", "gold", "pred_id", "pred", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    experiment_name: str,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": experiment_name,
            "config": config,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dev_metrics": metrics,
        },
        path,
    )


RobertaERCClassifier = TransformerERCClassifier  # 兼容之前写好的 ensemble import


def build_optimizer(model: TransformerERCClassifier, encoder_lr: float, classifier_lr: float, weight_decay: float):
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.classifier.parameters(), "lr": classifier_lr},
        ],
        weight_decay=weight_decay,
    )


def train_model(args: argparse.Namespace, config: dict) -> None:
    seed = args.seed if args.seed is not None else int(config.get("seed", 114514))
    set_seed(seed)

    experiment_name = str(config.get("model_name", "text_finetune"))
    model_name = str(config.get("text_model", "FacebookAI/roberta-base"))
    local_files_only = bool(config.get("local_files_only", False))
    pooling = str(config.get("pooling", "cls"))
    max_length = int(config.get("max_length", 128))
    context_window = int(config.get("context_window", 0))
    include_speaker = bool(config.get("include_speaker", True))
    batch_size = args.batch_size or int(config.get("batch_size", 16))
    max_epochs = args.max_epochs or int(config.get("max_epochs", 8))
    patience = args.patience or int(config.get("early_stopping_patience", 3))
    num_classes = int(config.get("num_classes", 7))
    dropout = float(config.get("dropout", 0.2))
    encoder_lr = args.encoder_lr or float(config.get("encoder_lr", 2e-5))
    classifier_lr = args.classifier_lr or float(config.get("classifier_lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.01))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    device = choose_device(args.device)
    use_amp = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    output_dir = Path(args.output_dir or f"results/{experiment_name}")
    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/{experiment_name}")
    log_path = output_dir / "train_log.csv"
    best_path = checkpoint_dir / f"best_{experiment_name}.pt"
    last_path = checkpoint_dir / f"last_{experiment_name}.pt"

    tokenizer = load_tokenizer(model_name, config)
    train_loader = make_loader(
        "train",
        tokenizer,
        args.data_root,
        max_length,
        batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_samples=args.max_train_samples,
        context_window=context_window,
        include_speaker=include_speaker,
    )
    dev_loader = make_loader(
        "dev",
        tokenizer,
        args.data_root,
        max_length,
        batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.max_dev_samples,
        context_window=context_window,
        include_speaker=include_speaker,
    )
    test_loader = make_loader(
        "test",
        tokenizer,
        args.data_root,
        max_length,
        batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.max_test_samples,
        context_window=context_window,
        include_speaker=include_speaker,
    )

    model = TransformerERCClassifier(
        model_name,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
        local_files_only=local_files_only,
    ).to(device)
    use_class_weights = str(config.get("loss", "weighted_cross_entropy")) == "weighted_cross_entropy"
    weights = class_weights(train_loader.dataset.labels, num_classes).to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(model, encoder_lr, classifier_lr, weight_decay)

    update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = update_steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"model={experiment_name} encoder={model_name} pooling={pooling} seed={seed}")
    print(f"context_window={context_window} include_speaker={include_speaker} max_length={max_length}")
    print(f"device={device} amp={use_amp} train={len(train_loader.dataset)} dev={len(dev_loader.dataset)}")
    print(f"batch_size={batch_size} encoder_lr={encoder_lr} classifier_lr={classifier_lr}")
    print(f"loss={config.get('loss', 'weighted_cross_entropy')} class_weights={use_class_weights}")
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
            scheduler,
            device,
            grad_accum_steps,
            max_grad_norm,
            use_amp,
        )
        dev_metrics, _ = evaluate(model, dev_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        rows.append(row)
        save_epoch_log(log_path, rows)
        save_checkpoint(last_path, model, optimizer, config, experiment_name, epoch, dev_metrics)

        improved = dev_metrics["weighted_f1"] > best_weighted_f1
        if improved:
            best_weighted_f1 = dev_metrics["weighted_f1"]
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, config, experiment_name, epoch, dev_metrics)
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

    if args.eval_test or bool(config.get("eval_test", False)):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, prediction_rows = evaluate(model, test_loader, criterion, device)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(test_metrics, file, indent=2)
        save_predictions(output_dir / "test_predictions.csv", prediction_rows)
        print(
            f"test accuracy={test_metrics['accuracy']:.4f} "
            f"weighted_f1={test_metrics['weighted_f1']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )

    print(f"done. best_dev_weighted_f1={best_weighted_f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer text encoder on MELD.")
    parser.add_argument("--config", default="configs/text_finetune.yaml")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--encoder-lr", type=float)
    parser.add_argument("--classifier-lr", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--eval-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_model(args, config)


if __name__ == "__main__":
    main()
