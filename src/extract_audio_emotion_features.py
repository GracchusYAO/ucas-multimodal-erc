"""提取 SER 取向的音频情绪特征。

相比通用 HuBERT mean pooling，这里使用已经做过语音情绪识别微调的 Wav2Vec2。
输出特征由三部分拼接：

1. hidden mean：模型最后一层语音表示的时间平均。
2. emotion logits：SER 模型自己的情绪分类 logits。
3. emotion probabilities：logits softmax 后的情绪概率。

默认输出：
    features/audio_emotion/{split}.pt
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


DEFAULT_MODEL = "Dpngtm/wav2vec2-emotion-recognition"
DEFAULT_OUTPUT_DIR = "features/audio_emotion"


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
    model_name: str,
    label_names: list[str],
    sample_rate: int,
    max_audio_seconds: float,
) -> dict:
    return {
        "split": split,
        "model_name": model_name,
        "label_names": label_names,
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "feature_layout": "hidden_mean + emotion_logits + emotion_probs",
        "features": features,  # [样本数, hidden_dim + 2 * num_ser_labels]
        "available": available,  # False 表示音频解码失败，用零向量占位
        "failed_keys": failed_keys,
        "labels": torch.tensor([item.emotion_id for item in utterances], dtype=torch.long),
        "dialogue_ids": torch.tensor([item.dialogue_id for item in utterances], dtype=torch.long),
        "utterance_ids": torch.tensor([item.utterance_id for item in utterances], dtype=torch.long),
        "keys": [item.key for item in utterances],
        "media_paths": [str(item.media_path) for item in utterances],
        "emotions": [item.emotion for item in utterances],
        "sentiments": [item.sentiment for item in utterances],
    }


def masked_mean_hidden(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """对 Wav2Vec2 帧做 masked mean，避免 padding 进入句级特征。"""
    if attention_mask is None or not hasattr(model, "_get_feature_vector_attention_mask"):
        return hidden.mean(dim=1)

    frame_mask = model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
    frame_mask = frame_mask.to(hidden.device).unsqueeze(-1)
    return (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1)


def extract_split(
    split: str,
    data_root: str,
    output_dir: str,
    model_name: str,
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

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    device_obj = choose_device(device)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(device_obj)
    model.eval()

    hidden_dim = int(model.config.hidden_size)
    num_ser_labels = int(model.config.num_labels)
    feature_dim = hidden_dim + num_ser_labels * 2  # hidden + logits + probabilities
    id2label = getattr(model.config, "id2label", {}) or {}
    label_names = [str(id2label.get(index, index)) for index in range(num_ser_labels)]

    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)
    available = torch.zeros(len(utterances), dtype=torch.bool)
    failed_keys: list[str] = []

    for indices in tqdm(batch_indices(len(utterances), batch_size), desc=f"audio_emotion:{split}"):
        waveforms = []
        valid_indices = []

        for index in indices:
            item = utterances[index]
            waveform = read_audio_from_mp4(item.media_path, sample_rate, max_audio_seconds)
            if waveform is None or waveform.numel() == 0:
                failed_keys.append(item.key)
                continue
            waveforms.append(waveform.numpy())
            valid_indices.append(index)

        if not waveforms:
            continue

        encoded = extractor(
            waveforms,
            sampling_rate=sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device_obj) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # SER 微调后的最后一层语音表示
            pooled_hidden = masked_mean_hidden(model, hidden, encoded.get("attention_mask"))
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pooled = torch.cat([pooled_hidden, logits, probs], dim=1).cpu()

        features[valid_indices] = pooled
        available[valid_indices] = True

    payload = build_payload(
        split,
        utterances,
        features,
        available,
        failed_keys,
        model_name,
        label_names,
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
    parser = argparse.ArgumentParser(description="Extract SER audio-emotion features.")
    parser.add_argument("--config")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--model-name")
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

    model_name = args.model_name or config.get("audio_emotion_model", DEFAULT_MODEL)
    sample_rate = args.sample_rate or int(config.get("sample_rate", 16000))
    batch_size = args.batch_size or int(config.get("batch_size_audio_emotion", 4))
    max_audio_seconds = args.max_audio_seconds or float(config.get("max_audio_seconds", 12))

    for split in args.split or MELD_SPLITS:
        extract_split(
            split,
            args.data_root,
            args.output_dir,
            model_name,
            sample_rate,
            batch_size,
            max_audio_seconds,
            args.device,
            args.force,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
