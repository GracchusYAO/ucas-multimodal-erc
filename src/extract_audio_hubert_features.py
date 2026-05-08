"""提取 HuBERT 音频特征。

这个脚本不会覆盖原来的 Wav2Vec2 特征，而是输出到：
    features/audio_hubert/{split}.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from src.torch_import_patch import patch_inspect_for_torch, restore_common_builtins

restore_common_builtins()
patch_inspect_for_torch()

import torch

restore_common_builtins()

import yaml

restore_common_builtins()

from tqdm import tqdm

from src.dataset import MELD_SPLITS, MELDUtterance, load_meld_split, validate_split
from src.extract_audio_features import batch_indices, choose_device, read_audio_from_mp4


DEFAULT_OUTPUT_DIR = "features/audio_hubert"


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_payload(
    split: str,
    utterances: list[MELDUtterance],
    features: torch.Tensor,
    available: torch.Tensor,
    failed_keys: list[str],
    sample_rate: int,
    max_audio_seconds: float,
) -> dict:
    return {
        "split": split,
        "model_name": "torchaudio.pipelines.HUBERT_BASE",
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "features": features,  # [样本数, 768]
        "available": available,  # False 表示这条音频不可用，用零向量占位
        "failed_keys": failed_keys,
        "labels": torch.tensor([item.emotion_id for item in utterances], dtype=torch.long),
        "dialogue_ids": torch.tensor([item.dialogue_id for item in utterances], dtype=torch.long),
        "utterance_ids": torch.tensor([item.utterance_id for item in utterances], dtype=torch.long),
        "keys": [item.key for item in utterances],
        "media_paths": [str(item.media_path) for item in utterances],
        "emotions": [item.emotion for item in utterances],
        "sentiments": [item.sentiment for item in utterances],
    }


def mean_pool_valid_frames(hidden: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """按 HuBERT 输出长度做 mean pooling，避免 padding 帧混进来。"""
    if lengths is None:
        return hidden.mean(dim=1)

    frame_ids = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
    mask = frame_ids < lengths.unsqueeze(1)
    masked = hidden * mask.unsqueeze(-1)
    return masked.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)


def extract_split(
    split: str,
    data_root: str,
    output_dir: str,
    sample_rate: int,
    batch_size: int,
    max_audio_seconds: float,
    device: str | None,
    force: bool,
    dry_run: bool,
) -> Path:
    split = validate_split(split)
    utterances = load_meld_split(split, data_root)
    output_path = Path(output_dir) / f"{split}.pt"
    missing = [item for item in utterances if not item.media_exists]

    if dry_run:
        print(
            f"[dry-run] {split}: {len(utterances)} rows, "
            f"missing_media={len(missing)} -> {output_path}"
        )
        return output_path
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Use --force to overwrite.")

    import torchaudio

    device_obj = choose_device(device)
    bundle = torchaudio.pipelines.HUBERT_BASE
    if sample_rate != bundle.sample_rate:
        raise ValueError(f"HuBERT expects {bundle.sample_rate} Hz audio, got {sample_rate}.")

    model = bundle.get_model().to(device_obj)
    model.eval()

    feature_dim = int(bundle._params["encoder_embed_dim"])  # HUBERT_BASE 是 768
    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)
    available = torch.zeros(len(utterances), dtype=torch.bool)
    failed_keys: list[str] = []

    for indices in tqdm(batch_indices(len(utterances), batch_size), desc=f"audio_hubert:{split}"):
        waveforms = []
        valid_indices = []

        for index in indices:
            item = utterances[index]
            waveform = read_audio_from_mp4(item.media_path, sample_rate, max_audio_seconds)
            if waveform is None or waveform.numel() == 0:
                failed_keys.append(item.key)
                continue
            waveforms.append(waveform)
            valid_indices.append(index)

        if not waveforms:
            continue

        lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        padded = padded.to(device_obj)
        lengths = lengths.to(device_obj)

        with torch.no_grad():
            layer_outputs, output_lengths = model.extract_features(padded, lengths=lengths)
            hidden = layer_outputs[-1]  # 最后一层 HuBERT 表示
            pooled = mean_pool_valid_frames(hidden, output_lengths).cpu()

        features[valid_indices] = pooled
        available[valid_indices] = True

    payload = build_payload(
        split,
        utterances,
        features,
        available,
        failed_keys,
        sample_rate,
        max_audio_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"saved {output_path} {tuple(features.shape)} available={int(available.sum())}")
    if failed_keys:
        print(f"failed audio examples: {failed_keys[:5]}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HuBERT audio features.")
    parser.add_argument("--config")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-audio-seconds", type=float)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    sample_rate = args.sample_rate or int(config.get("sample_rate", 16000))
    batch_size = args.batch_size or int(config.get("batch_size_audio_hubert", 8))
    max_audio_seconds = args.max_audio_seconds or float(config.get("max_audio_seconds", 12))

    for split in args.split or MELD_SPLITS:
        extract_split(
            split,
            args.data_root,
            args.output_dir,
            sample_rate,
            batch_size,
            max_audio_seconds,
            args.device,
            args.force,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
