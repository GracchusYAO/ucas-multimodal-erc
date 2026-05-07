"""提取 RoBERTa 文本特征。

输出：
    features/text_roberta/{split}.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.dataset import MELD_SPLITS, MELDUtterance, load_meld_split, validate_split


DEFAULT_MODEL = "FacebookAI/roberta-base"
DEFAULT_OUTPUT_DIR = "features/text_roberta"


def load_config(path: str | None) -> dict:
    """读取 YAML 配置。没有配置文件时返回空 dict。"""
    if path is None:  # 允许不用配置文件，直接走默认参数
        return {}
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(device: str | None) -> torch.device:
    """优先用用户指定设备；没指定时自动选 cuda/cpu。"""
    if device:  # 命令行显式传 --device 时优先听命令行
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_items(items: list[MELDUtterance], batch_size: int) -> list[list[MELDUtterance]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """对非 padding token 做平均，得到一句话一个向量。"""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # padding 位置为 0
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # 只平均真实 token


def pool_text(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "cls":
        return hidden_states[:, 0]  # RoBERTa 的第一个 token 当整句向量
    if pooling == "mean":
        return mean_pool(hidden_states, attention_mask)  # 更稳的句向量做法
    raise ValueError(f"Unsupported text pooling: {pooling}")


def build_payload(
    split: str,
    utterances: list[MELDUtterance],
    features: torch.Tensor,
    model_name: str,
    pooling: str,
    max_length: int,
) -> dict:
    """保存特征时顺手保存标签和 id，训练脚本就不用再读一遍 CSV。"""
    return {
        "split": split,
        "model_name": model_name,
        "pooling": pooling,
        "max_length": max_length,
        "features": features,  # [样本数, 768]
        "labels": torch.tensor([item.emotion_id for item in utterances], dtype=torch.long),  # 训练标签
        "dialogue_ids": torch.tensor(
            [item.dialogue_id for item in utterances],
            dtype=torch.long,
        ),
        "utterance_ids": torch.tensor(
            [item.utterance_id for item in utterances],
            dtype=torch.long,
        ),
        "keys": [item.key for item in utterances],
        "texts": [item.text for item in utterances],
        "speakers": [item.speaker for item in utterances],
        "emotions": [item.emotion for item in utterances],
        "sentiments": [item.sentiment for item in utterances],
    }


def extract_split(
    split: str,
    data_root: str,
    output_dir: str,
    model_name: str,
    pooling: str,
    max_length: int,
    batch_size: int,
    device: str | None,
    force: bool,
    dry_run: bool,
) -> Path:
    split = validate_split(split)
    utterances = load_meld_split(split, data_root)  # 顺序必须和 CSV 保持一致
    output_path = Path(output_dir) / f"{split}.pt"  # 每个 split 单独缓存一个文件

    if dry_run:
        print(f"[dry-run] {split}: {len(utterances)} rows -> {output_path}")
        return output_path
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Use --force to overwrite.")

    # 真正开始抽特征时才加载 transformers；dry-run 不需要加载大模型依赖。
    from transformers import AutoModel, AutoTokenizer

    device_obj = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 文本转 token id
    model = AutoModel.from_pretrained(model_name).to(device_obj)  # 冻结 RoBERTa 只做特征抽取
    model.eval()  # 关闭 dropout，保证抽出来的特征稳定

    all_features: list[torch.Tensor] = []
    for batch in tqdm(batch_items(utterances, batch_size), desc=f"text:{split}"):
        # tokenizer 会把不同长度的句子 padding 到同一长度。
        encoded = tokenizer(
            [item.text for item in batch],
            padding=True,  # 同一 batch 内补齐长度
            truncation=True,  # 太长的句子截断到 max_length
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device_obj) for key, value in encoded.items()}  # 数据搬到 GPU/CPU

        with torch.no_grad():
            outputs = model(**encoded)  # last_hidden_state: [B, T, 768]
            pooled = pool_text(outputs.last_hidden_state, encoded["attention_mask"], pooling)
        all_features.append(pooled.cpu())  # 存回 CPU，避免显存越积越多

    features = torch.cat(all_features, dim=0)  # 拼成 [总样本数, 768]
    payload = build_payload(split, utterances, features, model_name, pooling, max_length)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)  # 保存特征、标签、dialogue_id 等训练所需信息
    print(f"saved {output_path} {tuple(features.shape)}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RoBERTa text features.")
    parser.add_argument("--config", default="configs/text_only.yaml")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--model-name")
    parser.add_argument("--pooling", choices=("mean", "cls"))
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = args.model_name or config.get("text_model", DEFAULT_MODEL)  # 命令行优先，其次配置，最后默认
    pooling = args.pooling or config.get("text_pooling", "mean")
    max_length = args.max_length or int(config.get("max_length", 128))
    batch_size = args.batch_size or int(config.get("batch_size_text", 32))

    for split in args.split or MELD_SPLITS:
        extract_split(
            split,
            args.data_root,
            args.output_dir,
            model_name,
            pooling,
            max_length,
            batch_size,
            args.device,
            args.force,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
