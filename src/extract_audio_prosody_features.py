"""提取更偏情绪相关的音频韵律/声学特征。

HuBERT 更像“语义音频表示”，但情绪识别还很依赖语速、能量、音高和频谱变化。
这个脚本从 mp4 音轨中抽取 MFCC、RMS、ZCR、spectral centroid、pitch 等统计量，
并用 train split 的均值/方差做标准化，输出：
    features/audio_prosody/{split}.pt
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

import torchaudio
import yaml
from tqdm import tqdm

from src.dataset import MELD_SPLITS, MELDUtterance, load_meld_split, validate_split
from src.extract_audio_features import read_audio_from_mp4


DEFAULT_OUTPUT_DIR = "features/audio_prosody"
STAT_NAMES = ("mean", "std", "min", "max")


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def stat_names(prefix: str, count: int | None = None) -> list[str]:
    """生成特征名，方便后面写报告时解释每一维来自哪里。"""
    if count is None:
        return [f"{prefix}_{name}" for name in STAT_NAMES]
    return [f"{prefix}_{index:02d}_{name}" for index in range(count) for name in STAT_NAMES]


def feature_names(n_mfcc: int, use_pitch: bool) -> list[str]:
    names = []
    names.extend(stat_names("mfcc", n_mfcc))
    for prefix in ("rms", "zcr", "centroid", "bandwidth", "rolloff", "flatness"):
        names.extend(stat_names(prefix))
    if use_pitch:
        names.extend(stat_names("pitch"))
        names.extend(stat_names("pitch_delta"))
    names.extend(["duration_seconds", "peak_abs", "silence_ratio"])
    return names


def summarize(values: torch.Tensor) -> torch.Tensor:
    """把一段帧级序列压成 mean/std/min/max 四个统计量。"""
    values = values.float().flatten()
    if values.numel() == 0:
        return torch.zeros(4)
    return torch.tensor(
        [
            float(values.mean()),
            float(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
            float(values.min()),
            float(values.max()),
        ],
        dtype=torch.float32,
    )


def make_frames(waveform: torch.Tensor, win_length: int, hop_length: int) -> torch.Tensor:
    """切成短帧；过短 utterance 先补零，避免 unfold 报错。"""
    if waveform.numel() < win_length:
        waveform = torch.nn.functional.pad(waveform, (0, win_length - waveform.numel()))
    return waveform.unfold(0, win_length, hop_length)  # [num_frames, win_length]


def frame_time_features(waveform: torch.Tensor, win_length: int, hop_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    frames = make_frames(waveform, win_length, hop_length)
    rms = torch.sqrt((frames * frames).mean(dim=1).clamp_min(1e-12))  # 帧能量
    signs = torch.sign(frames)
    zcr = (signs[:, 1:] * signs[:, :-1] < 0).float().mean(dim=1)  # 过零率，粗略反映清浊/噪声变化
    return rms, zcr


def spectral_features(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算频谱中心、频谱带宽、rolloff 和 flatness。"""
    if waveform.numel() < win_length:
        waveform = torch.nn.functional.pad(waveform, (0, win_length - waveform.numel()))
    window = torch.hann_window(win_length)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = spectrum.abs().clamp_min(1e-8)  # [freq_bins, frames]
    freqs = torch.linspace(0.0, sample_rate / 2.0, magnitude.size(0))
    mag_sum = magnitude.sum(dim=0).clamp_min(1e-8)

    centroid_hz = (magnitude * freqs[:, None]).sum(dim=0) / mag_sum
    bandwidth_hz = torch.sqrt(
        ((freqs[:, None] - centroid_hz[None, :]).pow(2) * magnitude).sum(dim=0) / mag_sum
    )
    threshold = 0.85 * mag_sum
    rolloff_index = (magnitude.cumsum(dim=0) >= threshold[None, :]).float().argmax(dim=0)
    rolloff_hz = freqs[rolloff_index]
    flatness = torch.exp(magnitude.log().mean(dim=0)) / magnitude.mean(dim=0).clamp_min(1e-8)

    nyquist = sample_rate / 2.0
    return centroid_hz / nyquist, bandwidth_hz / nyquist, rolloff_hz / nyquist, flatness.clamp(0.0, 1.0)


def pitch_features(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    """粗略 F0 统计；pitch 对语气/情绪通常比纯语义音频更敏感。"""
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform.unsqueeze(0),
            sample_rate,
            frame_time=0.02,
            win_length=30,
            freq_low=60,
            freq_high=600,
        ).squeeze(0)
    except Exception:
        pitch = torch.zeros(0)

    voiced = pitch[pitch > 1.0] / 600.0  # 缩放到比较稳定的数值范围
    if voiced.numel() > 1:
        delta = voiced[1:] - voiced[:-1]
    else:
        delta = torch.zeros(0)
    return voiced, delta


def mfcc_features(
    waveform: torch.Tensor,
    mfcc_transform: torchaudio.transforms.MFCC,
    n_mfcc: int,
) -> torch.Tensor:
    """每个 MFCC 系数取 mean/std/min/max。"""
    mfcc = mfcc_transform(waveform.unsqueeze(0)).squeeze(0)  # [n_mfcc, frames]
    pieces = []
    for index in range(n_mfcc):
        pieces.append(summarize(mfcc[index] / 40.0))  # 简单缩放，避免 MFCC 数值远大于其他特征
    return torch.cat(pieces)


def extract_one(
    waveform: torch.Tensor,
    sample_rate: int,
    max_audio_seconds: float,
    mfcc_transform: torchaudio.transforms.MFCC,
    n_mfcc: int,
    use_pitch: bool,
) -> torch.Tensor:
    """从一条 utterance 的 waveform 中抽取固定长度向量。"""
    win_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    n_fft = 512

    waveform = waveform.float()
    waveform = waveform - waveform.mean()  # 去掉直流偏移

    pieces = [mfcc_features(waveform, mfcc_transform, n_mfcc)]

    rms, zcr = frame_time_features(waveform, win_length, hop_length)
    centroid, bandwidth, rolloff, flatness = spectral_features(waveform, sample_rate, n_fft, win_length, hop_length)
    for values in (rms, zcr, centroid, bandwidth, rolloff, flatness):
        pieces.append(summarize(values))

    if use_pitch:
        pitch, delta = pitch_features(waveform, sample_rate)
        pieces.append(summarize(pitch))
        pieces.append(summarize(delta))

    duration = min(float(waveform.numel()) / sample_rate, max_audio_seconds)
    peak_abs = float(waveform.abs().max()) if waveform.numel() else 0.0
    silence_ratio = float((rms < 0.01).float().mean()) if rms.numel() else 1.0
    pieces.append(torch.tensor([duration, peak_abs, silence_ratio], dtype=torch.float32))

    return torch.cat(pieces)


def build_payload(
    split: str,
    utterances: list[MELDUtterance],
    features: torch.Tensor,
    available: torch.Tensor,
    failed_keys: list[str],
    feature_name_list: list[str],
    sample_rate: int,
    max_audio_seconds: float,
    n_mfcc: int,
    use_pitch: bool,
    standardization: dict[str, torch.Tensor] | None = None,
) -> dict:
    payload = {
        "split": split,
        "model_name": "mfcc_prosody_spectral_stats",
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "n_mfcc": n_mfcc,
        "use_pitch": use_pitch,
        "feature_names": feature_name_list,
        "features": features,
        "available": available,
        "failed_keys": failed_keys,
        "labels": torch.tensor([item.emotion_id for item in utterances], dtype=torch.long),
        "dialogue_ids": torch.tensor([item.dialogue_id for item in utterances], dtype=torch.long),
        "utterance_ids": torch.tensor([item.utterance_id for item in utterances], dtype=torch.long),
        "keys": [item.key for item in utterances],
        "media_paths": [str(item.media_path) for item in utterances],
        "emotions": [item.emotion for item in utterances],
        "sentiments": [item.sentiment for item in utterances],
    }
    if standardization is not None:
        payload["standardization"] = standardization
    return payload


def extract_raw_split(
    split: str,
    data_root: str,
    sample_rate: int,
    max_audio_seconds: float,
    n_mfcc: int,
    use_pitch: bool,
    dry_run: bool,
) -> dict:
    split = validate_split(split)
    utterances = load_meld_split(split, data_root)
    feature_name_list = feature_names(n_mfcc, use_pitch)
    feature_dim = len(feature_name_list)
    missing = [item for item in utterances if not item.media_exists]

    if dry_run:
        print(f"[dry-run] {split}: {len(utterances)} rows, missing_media={len(missing)}, dim={feature_dim}")

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 512,
            "win_length": int(0.025 * sample_rate),
            "hop_length": int(0.010 * sample_rate),
            "n_mels": 40,
            "center": True,
        },
    )

    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)
    available = torch.zeros(len(utterances), dtype=torch.bool)
    failed_keys: list[str] = []

    if dry_run:
        return build_payload(
            split,
            utterances,
            features,
            available,
            failed_keys,
            feature_name_list,
            sample_rate,
            max_audio_seconds,
            n_mfcc,
            use_pitch,
        )

    for index, item in tqdm(list(enumerate(utterances)), desc=f"audio_prosody:{split}"):
        waveform = read_audio_from_mp4(item.media_path, sample_rate, max_audio_seconds)
        if waveform is None or waveform.numel() == 0:
            failed_keys.append(item.key)
            continue
        features[index] = extract_one(
            waveform,
            sample_rate,
            max_audio_seconds,
            mfcc_transform,
            n_mfcc,
            use_pitch,
        )
        available[index] = True

    return build_payload(
        split,
        utterances,
        features,
        available,
        failed_keys,
        feature_name_list,
        sample_rate,
        max_audio_seconds,
        n_mfcc,
        use_pitch,
    )


def standardize_payloads(payloads: dict[str, dict]) -> dict[str, dict]:
    """用 train 的统计量标准化所有 split，避免 dev/test 信息泄漏。"""
    reference_split = "train" if "train" in payloads else next(iter(payloads))
    reference = payloads[reference_split]
    train_features = reference["features"]
    train_available = reference["available"]
    if train_available.any():
        train_features = train_features[train_available]

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardization = {"source_split": reference_split, "mean": mean, "std": std}

    for payload in payloads.values():
        payload["features"] = (payload["features"] - mean) / std
        payload["features"][~payload["available"]] = 0.0  # 缺失样本保持零向量
        payload["standardization"] = standardization
    return payloads


def save_payloads(payloads: dict[str, dict], output_dir: str, force: bool) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for split, payload in payloads.items():
        output_path = output_root / f"{split}.pt"
        if output_path.exists() and not force:
            raise FileExistsError(f"{output_path} already exists. Use --force to overwrite.")
        torch.save(payload, output_path)
        features = payload["features"]
        available = payload["available"]
        print(f"saved {output_path} {tuple(features.shape)} available={int(available.sum())}")
        if payload["failed_keys"]:
            print(f"failed audio examples: {payload['failed_keys'][:5]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract prosody/acoustic audio features.")
    parser.add_argument("--config")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--max-audio-seconds", type=float)
    parser.add_argument("--n-mfcc", type=int)
    parser.add_argument("--no-pitch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    sample_rate = args.sample_rate or int(config.get("sample_rate", 16000))
    max_audio_seconds = args.max_audio_seconds or float(config.get("max_audio_seconds", 12))
    n_mfcc = args.n_mfcc or int(config.get("n_mfcc", 20))
    use_pitch = not args.no_pitch and bool(config.get("use_pitch", True))

    splits = args.split or MELD_SPLITS
    payloads = {
        split: extract_raw_split(
            split,
            args.data_root,
            sample_rate,
            max_audio_seconds,
            n_mfcc,
            use_pitch,
            args.dry_run,
        )
        for split in splits
    }
    if args.dry_run:
        return
    if not args.dry_run:
        payloads = standardize_payloads(payloads)
    save_payloads(payloads, args.output_dir, args.force)


if __name__ == "__main__":
    main()
