"""提取 Wav2Vec2 音频特征。

MELD 的媒体文件是 mp4，所以这里先用 ffmpeg 从 mp4 里解出 16kHz 单声道音频，再送进 Wav2Vec2。缺失或解码失败的样本保留为零向量。

输出：
    features/audio_wav2vec2/{split}.pt
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import torch
import yaml
from tqdm import tqdm

from src.dataset import MELD_SPLITS, MELDUtterance, load_meld_split, validate_split


DEFAULT_MODEL = "facebook/wav2vec2-base"
DEFAULT_OUTPUT_DIR = "features/audio_wav2vec2"


def load_config(path: str | None) -> dict:
    if path is None:  # 没给 config 就全用默认值
        return {}
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(device: str | None) -> torch.device:
    if device:  # 例如 --device cuda
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_indices(length: int, batch_size: int) -> list[list[int]]:
    return [list(range(i, min(i + batch_size, length))) for i in range(0, length, batch_size)]


def read_audio_from_mp4(
    media_path: Path,
    sample_rate: int,
    max_seconds: float,
) -> torch.Tensor | None:
    """用 ffmpeg 把 mp4 音轨解码成 float32 waveform。"""
    if not media_path.exists():  # 官方缺失视频会走这里
        return None

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),  # 使用 imageio 自带 ffmpeg，少依赖系统环境
        "-v",
        "error",
        "-i",
        str(media_path),  # 输入 mp4
        "-t",
        str(max_seconds),  # 过长音频截断，避免显存/内存爆
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",  # 单声道
        "-ar",
        str(sample_rate),  # Wav2Vec2 常用 16kHz
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,  # ffmpeg 的原始 waveform 从 stdout 读
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return None

    if not result.stdout:
        return None
    return torch.frombuffer(bytearray(result.stdout), dtype=torch.float32).clone()  # 转成 1D waveform


def build_payload(
    split: str,
    utterances: list[MELDUtterance],
    features: torch.Tensor,
    available: torch.Tensor,
    failed_keys: list[str],
    model_name: str,
    sample_rate: int,
    max_audio_seconds: float,
) -> dict:
    return {
        "split": split,
        "model_name": model_name,
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "features": features,  # [样本数, 768]
        "available": available,  # False 表示这条音频不可用，用零向量占位
        "failed_keys": failed_keys,  # 记录具体失败样本，方便写报告
        "labels": torch.tensor([item.emotion_id for item in utterances], dtype=torch.long),
        "dialogue_ids": torch.tensor(
            [item.dialogue_id for item in utterances],
            dtype=torch.long,
        ),
        "utterance_ids": torch.tensor(
            [item.utterance_id for item in utterances],
            dtype=torch.long,
        ),
        "keys": [item.key for item in utterances],
        "media_paths": [str(item.media_path) for item in utterances],
        "emotions": [item.emotion for item in utterances],
        "sentiments": [item.sentiment for item in utterances],
    }


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
    utterances = load_meld_split(split, data_root)  # 按 CSV 原顺序读取
    output_path = Path(output_dir) / f"{split}.pt"
    missing = [item for item in utterances if not item.media_exists]  # dry-run 时展示缺文件数量

    if dry_run:
        print(
            f"[dry-run] {split}: {len(utterances)} rows, "
            f"missing_media={len(missing)} -> {output_path}"
        )
        return output_path
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Use --force to overwrite.")

    # 到这里才加载 Wav2Vec2，避免 dry-run 也触发模型依赖导入。
    from transformers import AutoModel, AutoProcessor

    device_obj = choose_device(device)
    processor = AutoProcessor.from_pretrained(model_name)  # waveform -> Wav2Vec2 输入
    model = AutoModel.from_pretrained(model_name).to(device_obj)  # 冻结 Wav2Vec2
    model.eval()  # 只抽特征，不训练

    # 先建零矩阵。读不到音频的样本保持零向量，available=False。
    feature_dim = int(model.config.hidden_size)  # wav2vec2-base 是 768
    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)  # 先全置零
    available = torch.zeros(len(utterances), dtype=torch.bool)  # 成功抽取后再改 True
    failed_keys: list[str] = []

    for indices in tqdm(batch_indices(len(utterances), batch_size), desc=f"audio:{split}"):
        waveforms = []  # 当前 batch 中真正读到的音频
        valid_indices = []  # waveforms 对应原始样本下标

        for index in indices:
            item = utterances[index]
            waveform = read_audio_from_mp4(item.media_path, sample_rate, max_audio_seconds)
            if waveform is None or waveform.numel() == 0:
                failed_keys.append(item.key)  # 保留失败记录，但不中断整个 split
                continue
            waveforms.append(waveform.numpy())  # processor 接收 numpy/list 更方便
            valid_indices.append(index)  # 后面写回 features 的位置

        if not waveforms:
            continue  # 整个 batch 都失败时直接跳过

        encoded = processor(
            waveforms,
            sampling_rate=sample_rate,
            padding=True,  # 同一 batch 内补齐音频长度
            return_tensors="pt",
        )
        encoded = {key: value.to(device_obj) for key, value in encoded.items()}  # 搬到 GPU/CPU

        with torch.no_grad():
            outputs = model(**encoded)  # [B, T, 768]
            pooled = outputs.last_hidden_state.mean(dim=1).cpu()  # 时间维平均成一句一个向量

        features[valid_indices] = pooled  # 写回原始样本位置，保持与 CSV 对齐
        available[valid_indices] = True  # 标记这几条音频有效

    payload = build_payload(
        split,
        utterances,
        features,
        available,
        failed_keys,
        model_name,
        sample_rate,
        max_audio_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)  # 保存特征和可用性 mask
    print(f"saved {output_path} {tuple(features.shape)} available={int(available.sum())}")
    if failed_keys:
        print(f"failed audio examples: {failed_keys[:5]}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 audio features.")
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

    model_name = args.model_name or config.get("audio_model", DEFAULT_MODEL)  # 命令行 > config > 默认值
    sample_rate = args.sample_rate or int(config.get("sample_rate", 16000))
    batch_size = args.batch_size or int(config.get("batch_size_audio", 4))
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
