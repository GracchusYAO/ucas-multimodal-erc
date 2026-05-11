"""逐个导出模型 logits，供后续离线融合使用。

之前混合 fine-tuned text 和 cached-feature multimodal 时，一次性加载很多大模型
容易让当前 WSL/conda 环境不稳定。这个脚本故意一次只加载一个 checkpoint，
把 dev/test 的 logits 存成小 `.pt` 文件，后续调融合权重时就不再碰大模型。
"""

from __future__ import annotations

import argparse
import os
import random
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

import yaml

from src.dataset import MELD_SPLITS
from src.feature_dataset import make_dialogue_loader, make_feature_loader
from src.models import build_model


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device: str | None) -> torch.device:
    """优先用用户指定设备；不指定时有 GPU 就用 GPU。"""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def active_modalities(config: dict) -> tuple[str, ...]:
    """本地读模态开关，避免 feature 导出时 import 文本微调脚本。"""
    enabled = config.get("modalities", {"text": True})
    names = (
        "text",
        "audio",
        "audio_hubert",
        "audio_hubert_stats",
        "audio_prosody",
        "audio_hubert_prosody",
        "visual",
        "visual_face",
    )
    return tuple(name for name in names if enabled.get(name, False))


def build_quality_tensor(batch: dict, modalities: tuple[str, ...], device: torch.device) -> torch.Tensor:
    """整理每个模态是否可用的标记；文本没有标记时默认可用。"""
    batch_size = batch["label"].size(0)
    quality = torch.ones(batch_size, len(modalities), device=device)
    for index, modality in enumerate(modalities):
        key = f"{modality}_available"
        if key in batch:
            quality[:, index] = batch[key].to(device).float()
    return quality


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
    """导出 cached-feature 模型 logits；这条路径不加载 transformers。"""
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


def export_one_split(args: argparse.Namespace, split: str, device: torch.device) -> Path:
    """导出单个 split；text 和 feature 模型分开走已有评估逻辑。"""
    if args.kind == "text":
        from src.evaluate_text_ensemble import collect_logits as collect_text_logits

        config = load_config(args.config)
        batch_size = args.batch_size or int(config.get("batch_size", 8))
        logits, labels, keys = collect_text_logits(
            config,
            args.checkpoint,
            split,
            args.data_root,
            batch_size,
            args.num_workers,
            device,
        )
    else:
        logits, labels, keys = collect_feature_logits(
            args.config,
            args.checkpoint,
            split,
            args.features_root,
            args.batch_size,
            args.num_workers,
            device,
        )

    output_dir = Path(args.output_root) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.pt"

    payload = {
        "name": args.name,
        "kind": args.kind,
        "split": split,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "logits": logits.cpu().float(),  # 只保留 CPU float logits，方便后续快速读取
        "labels": labels.cpu().long(),
        "keys": list(keys),  # key 用来检查不同模型的样本顺序是否完全一致
    }
    torch.save(payload, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dev/test logits from one checkpoint.")
    parser.add_argument("--kind", choices=("text", "feature"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--features-root", default="features")
    parser.add_argument("--output-root", default="results/logits")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=114514)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    splits = args.split or ["dev", "test"]

    print(f"exporting {args.kind} logits: name={args.name}, device={device}")
    for split in splits:
        path = export_one_split(args, split, device)
        print(f"saved {split}: {path}")


if __name__ == "__main__":
    main()
