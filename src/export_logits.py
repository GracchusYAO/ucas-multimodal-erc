"""逐个导出模型 logits，供后续离线融合使用。

之前混合 fine-tuned text 和 cached-feature multimodal 时，一次性加载很多大模型
容易让当前 WSL/conda 环境不稳定。这个脚本故意一次只加载一个 checkpoint，
把 dev/test 的 logits 存成小 `.pt` 文件，后续调融合权重时就不再碰大模型。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from src.dataset import MELD_SPLITS
from src.evaluate_mixed_ensemble import collect_feature_logits
from src.evaluate_text_ensemble import collect_logits as collect_text_logits
from src.evaluate_text_ensemble import load_config
from src.train_text_finetune import set_seed


def choose_device(device: str | None) -> torch.device:
    """优先用用户指定设备；不指定时有 GPU 就用 GPU。"""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_one_split(args: argparse.Namespace, split: str, device: torch.device) -> Path:
    """导出单个 split；text 和 feature 模型分开走已有评估逻辑。"""
    if args.kind == "text":
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
