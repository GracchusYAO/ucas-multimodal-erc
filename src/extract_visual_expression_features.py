"""提取 expression-oriented 视觉特征。

CLIP 更偏场景/语义，不是专门识别人脸表情。这个脚本先检测并裁剪人脸，
再使用 facial expression recognition 模型抽取表情 embedding 和 emotion logits。

默认模型：
    trpakov/vit-face-expression

输出：
    features/visual_expression/{split}.pt
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 本地下载模型时不用 xet，避免长时间无进度卡住

_original_metadata_version = importlib.metadata.version


def safe_metadata_version(package_name: str) -> str:
    """当前 conda 环境偶发读 package metadata 失败；这里只兜底 transformers 导入期会查的包。"""
    normalized = package_name.lower()
    if normalized == "torch":
        return "2.9.1"
    if normalized in {"scikit-learn", "sklearn"}:
        return "1.7.2"
    return _original_metadata_version(package_name)


importlib.metadata.version = safe_metadata_version

from src.torch_import_patch import patch_inspect_for_torch, restore_common_builtins, stub_torch_dynamo

restore_common_builtins()
patch_inspect_for_torch()

from transformers import AutoImageProcessor, AutoModelForImageClassification

import torch

restore_common_builtins()
stub_torch_dynamo(torch)

import cv2
import yaml
from tqdm import tqdm

from src.dataset import MELD_SPLITS, MELDUtterance, load_meld_split, validate_split
from src.extract_visual_face_features import crop_largest_face, load_face_detector, sample_frames


DEFAULT_MODEL = "trpakov/vit-face-expression"
DEFAULT_OUTPUT_DIR = "features/visual_expression"


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def choose_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_frame_features(model, inputs: dict) -> torch.Tensor:
    """返回每帧的 expression feature。

    优先使用最后一层 CLS hidden state 作为表情 embedding，再拼接 emotion logits 和概率。
    如果模型没有 hidden_states，就至少保留 logits/probabilities。
    """
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits.float()
    probs = torch.softmax(logits, dim=1)

    if getattr(outputs, "hidden_states", None):
        hidden = outputs.hidden_states[-1]
        if hidden.ndim == 3:
            embedding = hidden[:, 0]  # ViT CLS token，作为表情 embedding
        else:
            embedding = hidden.flatten(start_dim=1)
        return torch.cat([embedding.float(), logits, probs], dim=1)

    return torch.cat([logits, probs], dim=1)


def build_payload(
    split: str,
    utterances: list[MELDUtterance],
    features: torch.Tensor,
    available: torch.Tensor,
    frame_counts: torch.Tensor,
    face_counts: torch.Tensor,
    failed_keys: list[str],
    model_name: str,
    id2label: dict,
    num_frames: int,
    face_margin: float,
    face_only: bool,
    pooling: str,
    top_k_frames: int,
) -> dict:
    return {
        "split": split,
        "model_name": model_name,
        "id2label": id2label,
        "num_frames": num_frames,
        "face_margin": face_margin,
        "face_only": face_only,
        "pooling": pooling,
        "top_k_frames": top_k_frames,
        "features": features,
        "available": available,
        "frame_counts": frame_counts,
        "face_counts": face_counts,
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
    face_only: bool,
    pooling: str,
    top_k_frames: int,
    local_files_only: bool,
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

    device_obj = choose_device(device)
    detector = load_face_detector()
    print(f"loading expression processor: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    print(f"loading expression model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(model_name, local_files_only=local_files_only)
    model = model.to(device_obj)
    model.eval()

    id2label = {int(key): value for key, value in model.config.id2label.items()}

    feature_dim: int | None = None
    frame_features_by_sample: list[list[torch.Tensor]] = [[] for _ in utterances]
    frame_scores_by_sample: list[list[float]] = [[] for _ in utterances]
    frame_counts = torch.zeros(len(utterances), dtype=torch.long)
    face_counts = torch.zeros(len(utterances), dtype=torch.long)
    failed_keys: list[str] = []
    pending_frames = []
    pending_indices = []
    pending_has_face = []

    def flush() -> None:
        """批量跑表情模型，并暂存每帧特征，方便后面做 top-k pooling。"""
        nonlocal pending_frames, pending_indices, pending_has_face
        nonlocal feature_dim
        if not pending_frames:
            return

        inputs = processor(images=pending_frames, return_tensors="pt")
        inputs = {key: value.to(device_obj) for key, value in inputs.items()}
        with torch.no_grad():
            frame_features = extract_frame_features(model, inputs).cpu()

        feature_dim = int(frame_features.size(1))
        num_expression_labels = len(id2label)
        for sample_index, has_face, frame_feature in zip(pending_indices, pending_has_face, frame_features):
            # 最后 num_expression_labels 维是 softmax 概率；最大概率越高，说明这一帧表情判断越稳定。
            expression_probs = frame_feature[-num_expression_labels:]
            confidence = float(expression_probs.max().item())
            frame_features_by_sample[sample_index].append(frame_feature)
            frame_scores_by_sample[sample_index].append(confidence)
            frame_counts[sample_index] += 1
            face_counts[sample_index] += int(has_face)

        pending_frames = []
        pending_indices = []
        pending_has_face = []

    quiet_tqdm = bool(os.environ.get("TQDM_DISABLE"))
    for index, item in tqdm(
        list(enumerate(utterances)),
        desc=f"visual_expression:{split}",
        disable=quiet_tqdm,
    ):
        frames = sample_frames(item.media_path, num_frames)
        if not frames:
            failed_keys.append(item.key)
            continue

        used_frame_count = 0
        for frame in frames:
            crop, has_face = crop_largest_face(frame, detector, face_margin)
            if face_only and not has_face:
                continue  # 表情模型最好看人脸；face_only 时直接跳过无脸帧
            pending_frames.append(crop)
            pending_indices.append(index)
            pending_has_face.append(has_face)
            used_frame_count += 1

        if used_frame_count == 0:
            failed_keys.append(item.key)

        if len(pending_frames) >= batch_size:
            flush()

    flush()

    if feature_dim is None:
        raise RuntimeError("No visual expression features were extracted.")

    available = frame_counts > 0
    features = torch.zeros(len(utterances), feature_dim, dtype=torch.float32)
    for sample_index, sample_features in enumerate(frame_features_by_sample):
        if not sample_features:
            continue

        stacked = torch.stack(sample_features)
        if pooling == "topk_confident":
            scores = torch.tensor(frame_scores_by_sample[sample_index], dtype=torch.float32)
            k = min(top_k_frames, stacked.size(0))
            selected = scores.topk(k=k).indices
            features[sample_index] = stacked[selected].mean(dim=0)  # 只平均最可信的几帧
        else:
            features[sample_index] = stacked.mean(dim=0)

    payload = build_payload(
        split,
        utterances,
        features,
        available,
        frame_counts,
        face_counts,
        failed_keys,
        model_name,
        id2label,
        num_frames,
        face_margin,
        face_only,
        pooling,
        top_k_frames,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"saved {output_path} {tuple(features.shape)} "
        f"available={int(available.sum())} face_frames={int(face_counts.sum())} "
        f"pooling={pooling}"
    )
    if failed_keys:
        print(f"failed visual examples: {failed_keys[:5]}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract facial-expression visual features.")
    parser.add_argument("--config")
    parser.add_argument("--data-root", default="data/meld")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=MELD_SPLITS)
    parser.add_argument("--model-name")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--face-margin", type=float)
    parser.add_argument("--face-only", action="store_true")
    parser.add_argument("--pooling", choices=("mean", "topk_confident"))
    parser.add_argument("--top-k-frames", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_name = args.model_name or config.get("visual_expression_model", DEFAULT_MODEL)
    num_frames = args.num_frames or int(config.get("num_frames", 8))
    batch_size = args.batch_size or int(config.get("batch_size_visual", 32))
    face_margin = args.face_margin or float(config.get("face_margin", 0.35))
    face_only = args.face_only or bool(config.get("face_only", False))
    pooling = args.pooling or str(config.get("visual_expression_pooling", "mean"))
    top_k_frames = args.top_k_frames or int(config.get("top_k_frames", 4))

    for split in args.split or MELD_SPLITS:
        extract_split(
            split,
            args.data_root,
            args.output_dir,
            model_name,
            num_frames,
            batch_size,
            face_margin,
            face_only,
            pooling,
            top_k_frames,
            args.local_files_only,
            args.device,
            args.force,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
