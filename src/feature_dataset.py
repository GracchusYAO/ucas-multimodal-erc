"""读取已经缓存好的 MELD 特征。

训练模型时不再读原始 mp4，而是直接读这些缓存特征。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import MELD_SPLITS, validate_split


MODALITY_DIRS = {
    "text": "text_roberta",
    "audio": "audio_wav2vec2",
    "visual": "visual_clip",
}


def load_feature_payload(
    split: str,
    modality: str,
    features_root: str | Path = "features",
) -> dict:
    """读取某个 split、某个模态的缓存文件。"""
    split = validate_split(split)
    if modality not in MODALITY_DIRS:
        raise ValueError(f"Unknown modality: {modality}")

    path = Path(features_root) / MODALITY_DIRS[modality] / f"{split}.pt"  # 约定好的缓存路径
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)  # 本地缓存含字符串元数据


class CachedFeatureDataset(Dataset):
    """把 text/audio/visual 缓存特征包装成 PyTorch Dataset。"""

    def __init__(
        self,
        split: str,
        modalities: Sequence[str] = ("text",),
        features_root: str | Path = "features",
    ) -> None:
        self.split = validate_split(split)
        self.modalities = tuple(modalities)
        self.payloads = {
            modality: load_feature_payload(self.split, modality, features_root)
            for modality in self.modalities
        }

        reference = self.payloads[self.modalities[0]]  # 用第一个模态当对齐基准
        self.labels = reference["labels"].long()
        self.dialogue_ids = reference["dialogue_ids"].long()
        self.utterance_ids = reference["utterance_ids"].long()
        self.keys = reference["keys"]

        self._check_alignment()  # 确认不同模态样本顺序一致

    def _check_alignment(self) -> None:
        """检查各模态的 key/label 是否完全对齐。"""
        for modality, payload in self.payloads.items():
            if payload["keys"] != self.keys:
                raise ValueError(f"{modality} keys are not aligned with the reference.")
            if not torch.equal(payload["labels"].long(), self.labels):
                raise ValueError(f"{modality} labels are not aligned with the reference.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {
            "label": self.labels[index],  # 情绪类别 id
            "dialogue_id": self.dialogue_ids[index],  # 后面做对话上下文会用到
            "utterance_id": self.utterance_ids[index],
            "key": self.keys[index],  # 方便定位原始样本
        }

        for modality, payload in self.payloads.items():
            item[modality] = payload["features"][index].float()  # 单条样本的模态特征
            if "available" in payload:
                item[f"{modality}_available"] = payload["available"][index].bool()

        return item


class DialogueFeatureDataset(Dataset):
    """按 dialogue 分组后的缓存特征，给 BiGRU context 使用。"""

    def __init__(
        self,
        split: str,
        modalities: Sequence[str] = ("text", "audio", "visual"),
        features_root: str | Path = "features",
    ) -> None:
        self.base = CachedFeatureDataset(split, modalities, features_root)
        self.labels = self.base.labels  # 训练脚本算 class weights 时仍然用 utterance 级标签
        self.dialogue_indices = self._build_dialogue_indices()

    def _build_dialogue_indices(self) -> list[list[int]]:
        """把同一个 Dialogue_ID 的 utterance 下标放在一起。"""
        grouped: dict[int, list[int]] = {}
        for index, dialogue_id in enumerate(self.base.dialogue_ids.tolist()):
            grouped.setdefault(dialogue_id, []).append(index)

        dialogues = []
        for dialogue_id in sorted(grouped):
            indices = grouped[dialogue_id]
            indices.sort(key=lambda item: int(self.base.utterance_ids[item]))  # 保证对话内部顺序
            dialogues.append(indices)
        return dialogues

    def __len__(self) -> int:
        return len(self.dialogue_indices)

    def __getitem__(self, index: int) -> dict:
        indices = self.dialogue_indices[index]
        item = {
            "label": self.base.labels[indices],  # [L]
            "dialogue_id": self.base.dialogue_ids[indices],  # [L]
            "utterance_id": self.base.utterance_ids[indices],  # [L]
            "key": [self.base.keys[item] for item in indices],
        }

        for modality, payload in self.base.payloads.items():
            item[modality] = payload["features"][indices].float()  # [L, D]
            if "available" in payload:
                item[f"{modality}_available"] = payload["available"][indices].bool()

        return item


def collate_feature_batch(samples: list[dict]) -> dict:
    """DataLoader 的 batch 拼接函数。"""
    batch: dict = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)  # tensor 字段堆成 [B, ...]
        else:
            batch[key] = values  # key 这类字符串字段保留 list
    return batch


def collate_dialogue_batch(samples: list[dict]) -> dict:
    """把不同长度的 dialogue padding 到同一长度。"""
    batch: dict = {}
    max_len = max(sample["label"].size(0) for sample in samples)
    batch_size = len(samples)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    batch["key"] = [sample["key"] for sample in samples]

    for row, sample in enumerate(samples):
        length = sample["label"].size(0)
        mask[row, :length] = True

    batch["mask"] = mask  # True 表示真实 utterance，False 是 padding

    tensor_keys = [key for key, value in samples[0].items() if isinstance(value, torch.Tensor)]
    for key in tensor_keys:
        first = samples[0][key]
        if first.ndim == 1:
            padded = torch.zeros(batch_size, max_len, dtype=first.dtype)
        else:
            padded = torch.zeros(batch_size, max_len, first.size(1), dtype=first.dtype)

        for row, sample in enumerate(samples):
            length = sample[key].size(0)
            padded[row, :length] = sample[key]
        batch[key] = padded

    return batch


def make_feature_loader(
    split: str,
    modalities: Sequence[str] = ("text",),
    features_root: str | Path = "features",
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """创建训练/验证用的 DataLoader。"""
    dataset = CachedFeatureDataset(split, modalities, features_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # train 可以打乱，dev/test 不打乱
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
    )


def make_dialogue_loader(
    split: str,
    modalities: Sequence[str] = ("text", "audio", "visual"),
    features_root: str | Path = "features",
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """创建按 dialogue batching 的 DataLoader。"""
    dataset = DialogueFeatureDataset(split, modalities, features_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dialogue_batch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check cached MELD features.")
    parser.add_argument("--features-root", default="features")
    parser.add_argument("--split", default="train", choices=MELD_SPLITS)
    parser.add_argument("--modality", action="append", choices=tuple(MODALITY_DIRS), default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--by-dialogue", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modalities = tuple(args.modality or ["text"])
    if args.by_dialogue:
        loader = make_dialogue_loader(args.split, modalities, args.features_root, args.batch_size)
    else:
        loader = make_feature_loader(args.split, modalities, args.features_root, args.batch_size)
    batch = next(iter(loader))

    print(f"split={args.split}, modalities={modalities}, samples={len(loader.dataset)}")
    for modality in modalities:
        print(f"{modality}: {tuple(batch[modality].shape)}")
    print(f"labels: {tuple(batch['label'].shape)}")
    print(f"first_key: {batch['key'][0]}")


if __name__ == "__main__":
    main()
