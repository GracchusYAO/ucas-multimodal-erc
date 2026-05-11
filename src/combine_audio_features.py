"""拼接 HuBERT 与 prosody 音频特征。

输出的 audio_hubert_prosody 仍然是一条音频分支，只是把 HuBERT 的语义表示
和 prosody 的情绪相关统计拼到一起，供后续门控多模态模型使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset import MELD_SPLITS, validate_split


def load_payload(root: str | Path, split: str) -> dict:
    path = Path(root) / f"{validate_split(split)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def check_alignment(audio: dict, prosody: dict, split: str) -> None:
    """两个缓存必须和同一个 CSV 顺序完全对齐，才能直接 cat。"""
    if audio["keys"] != prosody["keys"]:
        raise ValueError(f"{split}: audio/prosody keys are not aligned.")
    if not torch.equal(audio["labels"].long(), prosody["labels"].long()):
        raise ValueError(f"{split}: audio/prosody labels are not aligned.")


def combine_split(audio_dir: str, prosody_dir: str, output_dir: str, split: str, force: bool) -> Path:
    split = validate_split(split)
    audio = load_payload(audio_dir, split)
    prosody = load_payload(prosody_dir, split)
    check_alignment(audio, prosody, split)

    audio_features = audio["features"].float()
    prosody_features = prosody["features"].float()
    features = torch.cat([audio_features, prosody_features], dim=1)  # HuBERT 语义 + prosody 情绪线索

    audio_available = audio.get("available", torch.ones(features.size(0), dtype=torch.bool)).bool()
    prosody_available = prosody.get("available", torch.ones(features.size(0), dtype=torch.bool)).bool()
    available = audio_available & prosody_available
    features[~available] = 0.0  # 缺失音频仍然保持零向量占位

    payload = dict(audio)
    payload.update(
        {
            "split": split,
            "model_name": "hubert_plus_prosody",
            "source_audio_model": audio.get("model_name"),
            "source_prosody_model": prosody.get("model_name"),
            "audio_feature_dim": int(audio_features.size(1)),
            "prosody_feature_dim": int(prosody_features.size(1)),
            "feature_names_prosody": prosody.get("feature_names", []),
            "features": features,
            "available": available,
            "failed_keys": sorted(set(audio.get("failed_keys", [])) | set(prosody.get("failed_keys", []))),
        }
    )

    output_path = Path(output_dir) / f"{split}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Use --force to overwrite.")
    torch.save(payload, output_path)
    print(
        f"saved {output_path} {tuple(features.shape)} "
        f"audio_dim={audio_features.size(1)} prosody_dim={prosody_features.size(1)} "
        f"available={int(available.sum())}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine HuBERT and prosody audio features.")
    parser.add_argument("--audio-dir", default="features/audio_hubert")
    parser.add_argument("--prosody-dir", default="features/audio_prosody")
    parser.add_argument("--output-dir", default="features/audio_hubert_prosody")
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.split or MELD_SPLITS:
        combine_split(args.audio_dir, args.prosody_dir, args.output_dir, split, args.force)


if __name__ == "__main__":
    main()
