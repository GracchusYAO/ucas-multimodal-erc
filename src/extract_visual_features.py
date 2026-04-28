"""提取 CLIP 视觉特征。

每个 utterance 视频均匀采样若干帧；每帧过 CLIP image encoder；最后对帧特征
求平均，得到一个 utterance 级视觉向量。缺失视频保留为零向量。

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
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_frames(media_path: Path, num_frames: int) -> list:
    """从视频中均匀取 num_frames 帧，返回 RGB 图像列表。"""
    if not media_path.exists():
        return []

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        return []

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        if num_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            frame_indices = [
                round(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]

        frames = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        capture.release()


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
        "features": features,
        "available": available,
        "frame_counts": frame_counts,
        "failed_keys": failed_keys,
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

    # 到这里才加载 CLIP，dry-run 时只检查数据数量和输出路径。
    from transformers import CLIPModel, CLIPProcessor

    device_obj = choose_device(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device_obj)
    model.eval()

    feature_dim = int(model.config.projection_dim)
    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)
    frame_counts = torch.zeros(len(utterances), dtype=torch.long)
    failed_keys: list[str] = []

    pending_frames = []
    pending_indices = []

    def flush() -> None:
        """把暂存的帧批量送进 CLIP，减少一次一帧的开销。"""
        nonlocal pending_frames, pending_indices
        if not pending_frames:
            return

        inputs = processor(images=pending_frames, return_tensors="pt")
        inputs = {key: value.to(device_obj) for key, value in inputs.items()}
        with torch.no_grad():
            frame_features = model.get_image_features(**inputs).cpu()

        for sample_index, frame_feature in zip(pending_indices, frame_features):
            features[sample_index] += frame_feature
            frame_counts[sample_index] += 1

        pending_frames = []
        pending_indices = []

    for index, item in tqdm(list(enumerate(utterances)), desc=f"visual:{split}"):
        frames = sample_frames(item.media_path, num_frames)
        if not frames:
            failed_keys.append(item.key)
            continue

        pending_frames.extend(frames)
        pending_indices.extend([index] * len(frames))
        if len(pending_frames) >= batch_size:
            flush()

    flush()

    available = frame_counts > 0
    features = features / frame_counts.clamp(min=1).unsqueeze(1)

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
    torch.save(payload, output_path)
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

    model_name = args.model_name or config.get("visual_model", DEFAULT_MODEL)
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
