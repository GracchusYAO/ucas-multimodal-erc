"""提取 face-centered CLIP 视觉特征。

相比 `extract_visual_features.py` 的整帧均匀 CLIP，这里先用 OpenCV Haar cascade
找人脸，再裁剪最大人脸区域送入 CLIP。找不到脸时回退到整帧，保证每条视频都有机会产生特征。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from src.torch_import_patch import patch_inspect_for_torch, restore_common_builtins, stub_torch_dynamo

restore_common_builtins()
patch_inspect_for_torch()

import torch

restore_common_builtins()
stub_torch_dynamo(torch)

import cv2
import yaml
from tqdm import tqdm

restore_common_builtins()

from src.dataset import MELD_SPLITS, MELDUtterance, load_meld_split, validate_split


DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT_DIR = "features/visual_face_clip"


def load_face_detector():
    """加载 OpenCV 自带的人脸检测器，不需要额外下载权重。"""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load face cascade: {cascade_path}")
    return detector


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
            frame_indices = [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

        frames = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        capture.release()


def crop_largest_face(frame, detector, margin: float):
    """裁剪最大人脸；找不到脸就返回原图。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    if len(faces) == 0:
        return frame, False

    x, y, width, height = max(faces, key=lambda box: int(box[2] * box[3]))
    pad_x = int(width * margin)
    pad_y = int(height * margin)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + width + pad_x)
    y2 = min(frame.shape[0], y + height + pad_y)
    return frame[y1:y2, x1:x2], True


def build_payload(
    split: str,
    utterances: list[MELDUtterance],
    features: torch.Tensor,
    available: torch.Tensor,
    frame_counts: torch.Tensor,
    face_counts: torch.Tensor,
    failed_keys: list[str],
    model_name: str,
    num_frames: int,
    face_margin: float,
) -> dict:
    return {
        "split": split,
        "model_name": model_name,
        "num_frames": num_frames,
        "face_margin": face_margin,
        "features": features,  # [样本数, 512]
        "available": available,
        "frame_counts": frame_counts,  # 成功送入 CLIP 的帧数
        "face_counts": face_counts,  # 其中检测到脸的帧数
        "failed_keys": failed_keys,
        "labels": torch.tensor([item.emotion_id for item in utterances], dtype=torch.long),
        "dialogue_ids": torch.tensor([item.dialogue_id for item in utterances], dtype=torch.long),
        "utterance_ids": torch.tensor([item.utterance_id for item in utterances], dtype=torch.long),
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
    face_margin: float,
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

    restore_common_builtins()
    from transformers.models.clip.modeling_clip import CLIPModel
    from transformers.models.clip.processing_clip import CLIPProcessor
    restore_common_builtins()

    device_obj = choose_device(device)
    detector = load_face_detector()
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)  # 只读本地缓存，避免离线环境反复联网重试
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device_obj)  # 冻结 CLIP，只用来抽图像特征
    model.eval()

    feature_dim = int(model.config.projection_dim)
    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)
    frame_counts = torch.zeros(len(utterances), dtype=torch.long)
    face_counts = torch.zeros(len(utterances), dtype=torch.long)
    failed_keys: list[str] = []
    pending_frames = []
    pending_indices = []
    pending_has_face = []

    def flush() -> None:
        """批量跑 CLIP，并把同一 utterance 的多帧特征累加起来。"""
        nonlocal pending_frames, pending_indices, pending_has_face
        if not pending_frames:
            return
        inputs = processor(images=pending_frames, return_tensors="pt")
        inputs = {key: value.to(device_obj) for key, value in inputs.items()}
        with torch.no_grad():
            frame_features = model.get_image_features(**inputs).cpu()

        for sample_index, has_face, frame_feature in zip(pending_indices, pending_has_face, frame_features):
            features[sample_index] += frame_feature
            frame_counts[sample_index] += 1
            face_counts[sample_index] += int(has_face)

        pending_frames = []
        pending_indices = []
        pending_has_face = []

    for index, item in tqdm(list(enumerate(utterances)), desc=f"visual_face:{split}"):
        frames = sample_frames(item.media_path, num_frames)
        if not frames:
            failed_keys.append(item.key)
            continue

        for frame in frames:
            crop, has_face = crop_largest_face(frame, detector, face_margin)
            pending_frames.append(crop)
            pending_indices.append(index)
            pending_has_face.append(has_face)

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
        face_counts,
        failed_keys,
        model_name,
        num_frames,
        face_margin,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"saved {output_path} {tuple(features.shape)} "
        f"available={int(available.sum())} face_frames={int(face_counts.sum())}"
    )
    if failed_keys:
        print(f"failed visual examples: {failed_keys[:5]}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract face-centered CLIP visual features.")
    parser.add_argument("--config")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--model-name")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--face-margin", type=float)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = args.model_name or config.get("visual_model", DEFAULT_MODEL)
    num_frames = args.num_frames or int(config.get("num_frames", 8))
    batch_size = args.batch_size or int(config.get("batch_size_visual", 32))
    face_margin = args.face_margin or float(config.get("face_margin", 0.35))

    for split in args.split or MELD_SPLITS:
        extract_split(
            split,
            args.data_root,
            args.output_dir,
            model_name,
            num_frames,
            batch_size,
            face_margin,
            args.device,
            args.force,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
