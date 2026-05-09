"""Baseline models."""

from __future__ import annotations

import torch
from torch import nn


class SingleModalityClassifier(nn.Module):
    """单模态 baseline：某一种缓存特征 -> MLP -> 情绪分类。"""

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, projection_dim),  # 先把单模态特征压到较小空间
            nn.ReLU(),
            nn.Dropout(dropout),  # 防止小数据集上过拟合
            nn.Linear(projection_dim, num_classes),  # 输出 7 类情绪 logits
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """输入 feature shape: [batch_size, dim]；输出 logits shape: [batch_size, 7]。"""
        return self.classifier(feature.float())  # 确保输入是 float tensor


class TextOnlyClassifier(SingleModalityClassifier):
    """Text-only baseline：RoBERTa 特征 -> MLP -> 情绪分类。"""


class AudioOnlyClassifier(SingleModalityClassifier):
    """Audio-only baseline：Wav2Vec2 特征 -> MLP -> 情绪分类。"""


class AudioHubertOnlyClassifier(SingleModalityClassifier):
    """Audio-only baseline：HuBERT 特征 -> MLP -> 情绪分类。"""


class AudioHubertStatsOnlyClassifier(SingleModalityClassifier):
    """Audio-only baseline：HuBERT mean+std 特征 -> MLP -> 情绪分类。"""


class VisualOnlyClassifier(SingleModalityClassifier):
    """Visual-only baseline：CLIP 帧平均特征 -> MLP -> 情绪分类。"""


class VisualFaceOnlyClassifier(SingleModalityClassifier):
    """Visual-face baseline：face-centered CLIP 特征 -> MLP -> 情绪分类。"""


class ConcatClassifier(nn.Module):
    """Concat baseline：把若干模态特征拼接后直接分类。"""

    def __init__(
        self,
        input_dims: tuple[int, ...],
        projection_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        input_dim = sum(input_dims)  # 例如 text+audio+visual: 768+768+512=2048
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, projection_dim),  # 先把拼接后的大向量压到统一维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),  # 输出 7 类情绪 logits
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """输入若干模态特征，输出 logits shape: [batch_size, 7]。"""
        fused = torch.cat([feature.float() for feature in features], dim=1)  # [B, sum(dim)]
        return self.classifier(fused)


class TextAudioClassifier(ConcatClassifier):
    """Text+Audio concat baseline。"""


class TextAudioHubertClassifier(ConcatClassifier):
    """Text+HuBERT-Audio concat baseline。"""


class TextAudioHubertStatsClassifier(ConcatClassifier):
    """Text+HuBERT mean+std audio concat baseline。"""


class TextVisualClassifier(ConcatClassifier):
    """Text+Visual concat baseline。"""


class TextVisualFaceClassifier(ConcatClassifier):
    """Text+Face-Visual concat baseline。"""


class ConcatTAVClassifier(ConcatClassifier):
    """Text+Audio+Visual concat baseline。"""


def build_text_only_model(config: dict | None = None) -> TextOnlyClassifier:
    """按 YAML 配置构造 text-only baseline。"""
    config = config or {}
    return TextOnlyClassifier(
        input_dim=int(config.get("output_dim_text", 768)),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_audio_only_model(config: dict | None = None) -> AudioOnlyClassifier:
    """按 YAML 配置构造 audio-only baseline。"""
    config = config or {}
    return AudioOnlyClassifier(
        input_dim=int(config.get("output_dim_audio", 768)),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_audio_hubert_only_model(config: dict | None = None) -> AudioHubertOnlyClassifier:
    """按 YAML 配置构造 HuBERT audio-only baseline。"""
    config = config or {}
    return AudioHubertOnlyClassifier(
        input_dim=int(config.get("output_dim_audio_hubert", 768)),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_audio_hubert_stats_only_model(config: dict | None = None) -> AudioHubertStatsOnlyClassifier:
    """按 YAML 配置构造 HuBERT mean+std audio-only baseline。"""
    config = config or {}
    return AudioHubertStatsOnlyClassifier(
        input_dim=int(config.get("output_dim_audio_hubert_stats", config.get("output_dim_audio_hubert", 1536))),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_visual_only_model(config: dict | None = None) -> VisualOnlyClassifier:
    """按 YAML 配置构造 visual-only baseline。"""
    config = config or {}
    return VisualOnlyClassifier(
        input_dim=int(config.get("output_dim_visual", 512)),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_visual_face_only_model(config: dict | None = None) -> VisualFaceOnlyClassifier:
    """按 YAML 配置构造 face-centered visual-only baseline。"""
    config = config or {}
    return VisualFaceOnlyClassifier(
        input_dim=int(config.get("output_dim_visual_face", config.get("output_dim_visual", 512))),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_text_audio_model(config: dict | None = None) -> TextAudioClassifier:
    """按 YAML 配置构造 text+audio concat baseline。"""
    config = config or {}
    return TextAudioClassifier(
        input_dims=(
            int(config.get("output_dim_text", 768)),
            int(config.get("output_dim_audio", 768)),
        ),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_text_audio_hubert_model(config: dict | None = None) -> TextAudioHubertClassifier:
    """按 YAML 配置构造 text+HuBERT-audio concat baseline。"""
    config = config or {}
    return TextAudioHubertClassifier(
        input_dims=(
            int(config.get("output_dim_text", 768)),
            int(config.get("output_dim_audio_hubert", 768)),
        ),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_text_audio_hubert_stats_model(config: dict | None = None) -> TextAudioHubertStatsClassifier:
    """按 YAML 配置构造 text+HuBERT mean+std audio concat baseline。"""
    config = config or {}
    return TextAudioHubertStatsClassifier(
        input_dims=(
            int(config.get("output_dim_text", 768)),
            int(config.get("output_dim_audio_hubert_stats", config.get("output_dim_audio_hubert", 1536))),
        ),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_text_visual_model(config: dict | None = None) -> TextVisualClassifier:
    """按 YAML 配置构造 text+visual concat baseline。"""
    config = config or {}
    return TextVisualClassifier(
        input_dims=(
            int(config.get("output_dim_text", 768)),
            int(config.get("output_dim_visual", 512)),
        ),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_text_visual_face_model(config: dict | None = None) -> TextVisualFaceClassifier:
    """按 YAML 配置构造 text+face-visual concat baseline。"""
    config = config or {}
    return TextVisualFaceClassifier(
        input_dims=(
            int(config.get("output_dim_text", 768)),
            int(config.get("output_dim_visual_face", config.get("output_dim_visual", 512))),
        ),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_concat_tav_model(config: dict | None = None) -> ConcatTAVClassifier:
    """按 YAML 配置构造三模态拼接 baseline。"""
    config = config or {}
    modalities = config.get("modalities", {})

    # 这里根据配置自动选音频维度，方便比较普通 audio / HuBERT / HuBERT mean+std。
    if modalities.get("audio_hubert_stats", False):
        audio_dim = int(config.get("output_dim_audio_hubert_stats", 1536))
    elif modalities.get("audio_hubert", False):
        audio_dim = int(config.get("output_dim_audio_hubert", 768))
    else:
        audio_dim = int(config.get("output_dim_audio", 768))

    # 视觉同理：整帧 CLIP 和人脸裁剪 CLIP 都共用这个 concat baseline。
    if modalities.get("visual_face", False):
        visual_dim = int(config.get("output_dim_visual_face", config.get("output_dim_visual", 512)))
    else:
        visual_dim = int(config.get("output_dim_visual", 512))

    return ConcatTAVClassifier(
        input_dims=(
            int(config.get("output_dim_text", 768)),
            audio_dim,
            visual_dim,
        ),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_baseline_model(config: dict | None = None) -> nn.Module:
    """根据 model_name 选择 baseline 模型。"""
    config = config or {}
    model_name = config.get("model_name", "text_only")
    if model_name == "text_only":
        return build_text_only_model(config)
    if model_name == "audio_only":
        return build_audio_only_model(config)
    if model_name == "audio_hubert_only":
        return build_audio_hubert_only_model(config)
    if model_name == "audio_hubert_stats_only":
        return build_audio_hubert_stats_only_model(config)
    if model_name == "visual_only":
        return build_visual_only_model(config)
    if model_name == "visual_face_only":
        return build_visual_face_only_model(config)
    if model_name == "text_audio":
        return build_text_audio_model(config)
    if model_name == "text_audio_hubert":
        return build_text_audio_hubert_model(config)
    if model_name == "text_audio_hubert_stats":
        return build_text_audio_hubert_stats_model(config)
    if model_name == "text_visual":
        return build_text_visual_model(config)
    if model_name == "text_visual_face":
        return build_text_visual_face_model(config)
    if model_name in {"concat_tav", "concat_tav_hubert_stats_face"}:
        return build_concat_tav_model(config)
    raise ValueError(f"Unsupported baseline model: {model_name}")
