"""Baseline models."""

from __future__ import annotations

import torch
from torch import nn


class TextOnlyClassifier(nn.Module):
    """Text-only baseline：RoBERTa 特征 -> MLP -> 情绪分类。"""

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, projection_dim),  # 先把 RoBERTa 768 维压到较小空间
            nn.ReLU(),
            nn.Dropout(dropout),  # 防止小数据集上过拟合
            nn.Linear(projection_dim, num_classes),  # 输出 7 类情绪 logits
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """输入 text shape: [batch_size, 768]；输出 logits shape: [batch_size, 7]。"""
        return self.classifier(text.float())  # 确保输入是 float tensor


class ConcatTAVClassifier(nn.Module):
    """Text+Audio+Visual concat baseline：三种特征拼接后直接分类。"""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        input_dim = text_dim + audio_dim + visual_dim  # 768 + 768 + 512 = 2048
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, projection_dim),  # 先把拼接后的大向量压到统一维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),  # 输出 7 类情绪 logits
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        """输入三种模态特征，输出 logits shape: [batch_size, 7]。"""
        fused = torch.cat([text.float(), audio.float(), visual.float()], dim=1)  # [B, 2048]
        return self.classifier(fused)


def build_text_only_model(config: dict | None = None) -> TextOnlyClassifier:
    """按 YAML 配置构造 text-only baseline。"""
    config = config or {}
    return TextOnlyClassifier(
        input_dim=int(config.get("output_dim_text", 768)),
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.3)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_concat_tav_model(config: dict | None = None) -> ConcatTAVClassifier:
    """按 YAML 配置构造三模态拼接 baseline。"""
    config = config or {}
    return ConcatTAVClassifier(
        text_dim=int(config.get("output_dim_text", 768)),
        audio_dim=int(config.get("output_dim_audio", 768)),
        visual_dim=int(config.get("output_dim_visual", 512)),
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
    if model_name == "concat_tav":
        return build_concat_tav_model(config)
    raise ValueError(f"Unsupported baseline model: {model_name}")
