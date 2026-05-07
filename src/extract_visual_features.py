"""提取 CLIP 视觉特征。

每个 utterance 视频均匀采样若干帧；每帧过 CLIP image encoder；最后对帧特征求平均，得到一个 utterance 级视觉向量。缺失视频保留为零向量。

输出：
    features/visual_clip/{split}.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
import yaml
from tqdm import tqdm

from src.dataset import MELD_SPLITS, MELDUtterance, load_meld_split, validate_split


DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT_DIR = "features/visual_clip"


def load_config(path: str | None) -> dict:
    if path is None:  # 没给 config 就使用默认参数
        return {}
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(device: str | None) -> torch.device:
    if device:  # 例如 --device cuda
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_frames(media_path: Path, num_frames: int) -> list:
    """从视频中均匀取 num_frames 帧，返回 RGB 图像列表。"""
    if not media_path.exists():  # 缺失视频直接返回空帧列表
        return []

    capture = cv2.VideoCapture(str(media_path))  # OpenCV 负责读 mp4 帧
    if not capture.isOpened():
        return []

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
        if total_frames <= 0:
            return []

        if num_frames == 1:
            frame_indices = [total_frames // 2]  # 只取一帧时取中间帧
        else:
            frame_indices = [
                round(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]  # 均匀取 num_frames 个位置

        frames = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))  # 跳到目标帧
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV 默认 BGR，CLIP 需要 RGB
        return frames
    finally:
        capture.release()  # 无论成功失败都释放视频句柄


def build_payload(
    split: str,
    utterances: list[MELDUtterance],
    features: torch.Tensor,
    available: torch.Tensor,
    frame_counts: torch.Tensor,
    failed_keys: list[str],
    model_name: str,
    num_frames: int,
) -> dict:
    return {
        "split": split,
        "model_name": model_name,
        "num_frames": num_frames,
        "features": features,  # [样本数, 512]
        "available": available,  # False 表示该视频没读到帧
        "frame_counts": frame_counts,  # 每条样本实际成功读到几帧
        "failed_keys": failed_keys,  # 记录失败样本用于报告
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
    num_frames: int,
    batch_size: int,
    device: str | None,
    force: bool,
    dry_run: bool,
) -> Path:
    split = validate_split(split)
    utterances = load_meld_split(split, data_root)  # 顺序保持和 CSV 一致
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

    # 到这里才加载 CLIP，dry-run 时只检查数据数量和输出路径。
    from transformers import CLIPModel, CLIPProcessor

    device_obj = choose_device(device)
    processor = CLIPProcessor.from_pretrained(model_name)  # 图像预处理：resize/normalize
    model = CLIPModel.from_pretrained(model_name).to(device_obj)  # 冻结 CLIP image encoder
    model.eval()  # 只抽特征，不更新参数

    feature_dim = int(model.config.projection_dim)  # clip-vit-base-patch32 是 512
    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)  # 先全置零
    frame_counts = torch.zeros(len(utterances), dtype=torch.long)  # 后面用于求平均
    failed_keys: list[str] = []

    pending_frames = []  # 暂存还没送进 CLIP 的帧
    pending_indices = []  # 每一帧属于哪条 utterance

    def flush() -> None:
        """把暂存的帧批量送进 CLIP，减少一次一帧的开销。"""
        nonlocal pending_frames, pending_indices
        if not pending_frames:
            return

        inputs = processor(images=pending_frames, return_tensors="pt")  # 一批帧一起预处理
        inputs = {key: value.to(device_obj) for key, value in inputs.items()}  # 搬到 GPU/CPU
        with torch.no_grad():
            frame_features = model.get_image_features(**inputs).cpu()  # 每帧一个 512 维向量

        for sample_index, frame_feature in zip(pending_indices, frame_features):
            features[sample_index] += frame_feature  # 同一视频多帧先累加
            frame_counts[sample_index] += 1  # 记录累加了几帧

        pending_frames = []  # 清空暂存区，准备下一批
        pending_indices = []

    for index, item in tqdm(list(enumerate(utterances)), desc=f"visual:{split}"):
        frames = sample_frames(item.media_path, num_frames)
        if not frames:
            failed_keys.append(item.key)  # 失败样本保持零向量
            continue

        pending_frames.extend(frames)  # 一个视频会贡献多帧
        pending_indices.extend([index] * len(frames))  # 多帧都指向同一条样本
        if len(pending_frames) >= batch_size:
            flush()  # 凑够 batch_size 帧就跑一次 CLIP

    flush()  # 处理最后不足一个 batch 的剩余帧

    available = frame_counts > 0  # 至少读到一帧才算有效
    features = features / frame_counts.clamp(min=1).unsqueeze(1)  # 帧特征平均成视频特征

    payload = build_payload(
        split,
        utterances,
        features,
        available,
        frame_counts,
        failed_keys,
        model_name,
        num_frames,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)  # 保存视觉特征和 frame_counts
    print(f"saved {output_path} {tuple(features.shape)} available={int(available.sum())}")
    if failed_keys:
        print(f"failed visual examples: {failed_keys[:5]}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP visual features.")
    parser.add_argument("--config")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--model-name")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = args.model_name or config.get("visual_model", DEFAULT_MODEL)  # 命令行 > config > 默认值
    num_frames = args.num_frames or int(config.get("num_frames", 4))
    batch_size = args.batch_size or int(config.get("batch_size_visual", 16))

    for split in args.split or MELD_SPLITS:
        extract_split(
            split,
            args.data_root,
            args.output_dir,
            model_name,
            num_frames,
            batch_size,
            args.device,
            args.force,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
