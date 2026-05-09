"""Dynamic reliability-gated fusion model."""

from __future__ import annotations

import torch
from torch import nn


def make_projection(
    input_dim: int,
    d_model: int,
    dropout: float,
    use_layernorm: bool,
) -> nn.Sequential:
    """把不同模态特征投影到同一个维度。"""
    layers: list[nn.Module] = [
        nn.Linear(input_dim, d_model),  # 原始模态维度 -> 统一融合维度
        nn.ReLU(),
        nn.Dropout(dropout),
    ]
    if use_layernorm:
        layers.append(nn.LayerNorm(d_model))  # 让三个模态的数值尺度更接近
    return nn.Sequential(*layers)


class DynamicGatedFusionClassifier(nn.Module):
    """DGF：每条 utterance 动态计算 text/audio/visual 三个权重。"""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 512,
        d_model: int = 256,
        projection_dropout: float = 0.3,
        use_layernorm: bool = True,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
        num_classes: int = 7,
        use_modality_dropout: bool = False,
        modality_dropout_p: float = 0.3,
        drop_text_p: float = 0.1,
        drop_audio_p: float = 0.2,
        drop_visual_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.text_proj = make_projection(text_dim, d_model, projection_dropout, use_layernorm)
        self.audio_proj = make_projection(audio_dim, d_model, projection_dropout, use_layernorm)
        self.visual_proj = make_projection(visual_dim, d_model, projection_dropout, use_layernorm)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, gate_hidden_dim),  # 看三种模态后决定权重
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 3),  # 输出 text/audio/visual 三个 gate logit
        )
        self.classifier = nn.Linear(d_model, num_classes)

        self.use_modality_dropout = use_modality_dropout
        self.modality_dropout_p = modality_dropout_p
        self.drop_probs = torch.tensor([drop_text_p, drop_audio_p, drop_visual_p])

    def apply_modality_dropout(self, stacked: torch.Tensor) -> torch.Tensor:
        """训练时随机把某些模态置零，模拟模态缺失。"""
        if not self.training or not self.use_modality_dropout:
            return stacked

        batch_size = stacked.size(0)
        device = stacked.device
        apply_dropout = torch.rand(batch_size, 1, device=device) < self.modality_dropout_p
        drop_probs = self.drop_probs.to(device).view(1, 3)
        dropped = (torch.rand(batch_size, 3, device=device) < drop_probs) & apply_dropout

        all_dropped = dropped.all(dim=1)  # 避免一条样本三个模态全没了
        if all_dropped.any():
            dropped[all_dropped, 0] = False  # 至少保留 text

        keep = (~dropped).float().view(batch_size, 3, 1)
        return stacked * keep

    def fuse(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回融合向量和 gate 权重。"""
        z_text = self.text_proj(text.float())
        z_audio = self.audio_proj(audio.float())
        z_visual = self.visual_proj(visual.float())

        stacked = torch.stack([z_text, z_audio, z_visual], dim=1)  # [B, 3, d_model]
        stacked = self.apply_modality_dropout(stacked)

        gate_input = stacked.flatten(start_dim=1)  # [B, 3*d_model]
        gate_weights = torch.softmax(self.gate(gate_input), dim=1)  # [B, 3]
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # 加权求和
        return fused, gate_weights

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        return_gate: bool = False,
    ):
        fused, gate_weights = self.fuse(text, audio, visual)
        logits = self.classifier(fused)
        if return_gate:
            return logits, gate_weights
        return logits


class LateFusionHubertClassifier(nn.Module):
    """Late fusion：text / HuBERT-audio / visual 各自预测，再学习加权融合。"""

    def __init__(
        self,
        text_dim: int = 768,
        audio_hubert_dim: int = 768,
        visual_dim: int = 512,
        d_model: int = 256,
        projection_dropout: float = 0.3,
        use_layernorm: bool = True,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.text_proj = make_projection(text_dim, d_model, projection_dropout, use_layernorm)
        self.audio_proj = make_projection(audio_hubert_dim, d_model, projection_dropout, use_layernorm)
        self.visual_proj = make_projection(visual_dim, d_model, projection_dropout, use_layernorm)

        self.text_head = nn.Linear(d_model, num_classes)  # 文本单独给出一组 logits
        self.audio_head = nn.Linear(d_model, num_classes)  # HuBERT 音频单独给出一组 logits
        self.visual_head = nn.Linear(d_model, num_classes)  # CLIP 视觉单独给出一组 logits

        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, gate_hidden_dim),  # 根据三路表示判断每个模态可信度
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 3),
        )

    def forward(
        self,
        text: torch.Tensor,
        audio_hubert: torch.Tensor,
        visual: torch.Tensor,
        return_gate: bool = False,
    ):
        z_text = self.text_proj(text.float())
        z_audio = self.audio_proj(audio_hubert.float())
        z_visual = self.visual_proj(visual.float())

        logits = torch.stack(
            [
                self.text_head(z_text),
                self.audio_head(z_audio),
                self.visual_head(z_visual),
            ],
            dim=1,
        )  # [B, 3, num_classes]

        gate_input = torch.cat([z_text, z_audio, z_visual], dim=1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=1)  # [B, 3]
        fused_logits = (logits * gate_weights.unsqueeze(-1)).sum(dim=1)
        if return_gate:
            return fused_logits, gate_weights
        return fused_logits


class QualityLateFusionHubertClassifier(nn.Module):
    """Quality-aware late fusion：gate 同时看模态表示和可用性标记。"""

    uses_quality = True

    def __init__(
        self,
        text_dim: int = 768,
        audio_hubert_dim: int = 768,
        visual_dim: int = 512,
        d_model: int = 256,
        projection_dropout: float = 0.3,
        use_layernorm: bool = True,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
        num_classes: int = 7,
        text_gate_bias: float = 1.2,
        use_quality_dropout: bool = True,
        drop_text_p: float = 0.05,
        drop_audio_p: float = 0.25,
        drop_visual_p: float = 0.25,
    ) -> None:
        super().__init__()
        self.text_proj = make_projection(text_dim, d_model, projection_dropout, use_layernorm)
        self.audio_proj = make_projection(audio_hubert_dim, d_model, projection_dropout, use_layernorm)
        self.visual_proj = make_projection(visual_dim, d_model, projection_dropout, use_layernorm)

        self.text_head = nn.Linear(d_model, num_classes)
        self.audio_head = nn.Linear(d_model, num_classes)
        self.visual_head = nn.Linear(d_model, num_classes)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 3 + 3, gate_hidden_dim),  # 多拼 3 个 availability/quality 标记
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([text_gate_bias, 0.0, 0.0]))  # 初始更信文本

        self.use_quality_dropout = use_quality_dropout
        self.drop_probs = torch.tensor([drop_text_p, drop_audio_p, drop_visual_p])

    def apply_quality_dropout(self, quality: torch.Tensor) -> torch.Tensor:
        """训练时随机把某些模态标成不可用，逼 gate 学会处理缺失模态。"""
        if not self.training or not self.use_quality_dropout:
            return quality

        drop_probs = self.drop_probs.to(quality.device).view(1, 3)
        keep = torch.rand_like(quality) >= drop_probs
        keep = keep | (quality <= 0)  # 原本不可用的模态保持不可用

        all_dropped = (quality * keep.float()).sum(dim=1) <= 0
        if all_dropped.any():
            keep[all_dropped, 0] = True  # 至少保留文本，避免整条样本没有输入
        return quality * keep.float()

    def forward(
        self,
        text: torch.Tensor,
        audio_hubert: torch.Tensor,
        visual: torch.Tensor,
        quality: torch.Tensor | None = None,
        return_gate: bool = False,
    ):
        z_text = self.text_proj(text.float())
        z_audio = self.audio_proj(audio_hubert.float())
        z_visual = self.visual_proj(visual.float())

        if quality is None:
            quality = torch.ones(text.size(0), 3, device=text.device)
        quality = self.apply_quality_dropout(quality.float())

        logits = torch.stack(
            [
                self.text_head(z_text),
                self.audio_head(z_audio),
                self.visual_head(z_visual),
            ],
            dim=1,
        )

        gate_input = torch.cat([z_text, z_audio, z_visual, quality], dim=1)
        gate_logits = self.gate(gate_input)
        gate_logits = gate_logits.masked_fill(quality <= 0, -1e4)  # 不可用模态不参与加权
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused_logits = (logits * gate_weights.unsqueeze(-1)).sum(dim=1)
        if return_gate:
            return fused_logits, gate_weights
        return fused_logits


def build_dgf_model(config: dict | None = None) -> DynamicGatedFusionClassifier:
    """按 YAML 配置构造 DGF 模型。"""
    config = config or {}
    return DynamicGatedFusionClassifier(
        text_dim=int(config.get("output_dim_text", 768)),
        audio_dim=int(config.get("output_dim_audio", 768)),
        visual_dim=int(config.get("output_dim_visual", 512)),
        d_model=int(config.get("d_model", 256)),
        projection_dropout=float(config.get("projection_dropout", 0.3)),
        use_layernorm=bool(config.get("use_layernorm", True)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 128)),
        gate_dropout=float(config.get("gate_dropout", 0.2)),
        num_classes=int(config.get("num_classes", 7)),
        use_modality_dropout=bool(config.get("use_modality_dropout", False)),
        modality_dropout_p=float(config.get("modality_dropout_p", 0.3)),
        drop_text_p=float(config.get("drop_text_p", 0.1)),
        drop_audio_p=float(config.get("drop_audio_p", 0.2)),
        drop_visual_p=float(config.get("drop_visual_p", 0.2)),
    )


def build_late_fusion_hubert_model(config: dict | None = None) -> LateFusionHubertClassifier:
    """按 YAML 配置构造 text + HuBERT-audio + visual late fusion。"""
    config = config or {}
    return LateFusionHubertClassifier(
        text_dim=int(config.get("output_dim_text", 768)),
        audio_hubert_dim=int(config.get("output_dim_audio_hubert", 768)),
        visual_dim=int(config.get("output_dim_visual", 512)),
        d_model=int(config.get("d_model", 256)),
        projection_dropout=float(config.get("projection_dropout", 0.3)),
        use_layernorm=bool(config.get("use_layernorm", True)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 128)),
        gate_dropout=float(config.get("gate_dropout", 0.2)),
        num_classes=int(config.get("num_classes", 7)),
    )


def build_quality_late_fusion_hubert_model(config: dict | None = None) -> QualityLateFusionHubertClassifier:
    """按 YAML 配置构造 quality-aware late fusion。"""
    config = config or {}
    return QualityLateFusionHubertClassifier(
        text_dim=int(config.get("output_dim_text", 768)),
        audio_hubert_dim=int(config.get("output_dim_audio_hubert", 768)),
        visual_dim=int(config.get("output_dim_visual", 512)),
        d_model=int(config.get("d_model", 256)),
        projection_dropout=float(config.get("projection_dropout", 0.3)),
        use_layernorm=bool(config.get("use_layernorm", True)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 128)),
        gate_dropout=float(config.get("gate_dropout", 0.2)),
        num_classes=int(config.get("num_classes", 7)),
        text_gate_bias=float(config.get("text_gate_bias", 1.2)),
        use_quality_dropout=bool(config.get("use_quality_dropout", True)),
        drop_text_p=float(config.get("drop_text_p", 0.05)),
        drop_audio_p=float(config.get("drop_audio_p", 0.25)),
        drop_visual_p=float(config.get("drop_visual_p", 0.25)),
    )
