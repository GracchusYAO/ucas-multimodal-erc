"""Dialogue context encoder components."""

from __future__ import annotations

import torch
from torch import nn

from src.models.fusion import DynamicGatedFusionClassifier


class DGFContextClassifier(nn.Module):
    """DGF + BiGRU context：先融合每句话，再建模 dialogue 上下文。"""

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
        use_modality_dropout: bool = True,
        modality_dropout_p: float = 0.3,
        drop_text_p: float = 0.1,
        drop_audio_p: float = 0.2,
        drop_visual_p: float = 0.2,
        context_hidden_dim: int = 256,
        context_num_layers: int = 1,
        bidirectional: bool = True,
        context_dropout: float = 0.3,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fusion = DynamicGatedFusionClassifier(
            text_dim=text_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            d_model=d_model,
            projection_dropout=projection_dropout,
            use_layernorm=use_layernorm,
            gate_hidden_dim=gate_hidden_dim,
            gate_dropout=gate_dropout,
            num_classes=num_classes,
            use_modality_dropout=use_modality_dropout,
            modality_dropout_p=modality_dropout_p,
            drop_text_p=drop_text_p,
            drop_audio_p=drop_audio_p,
            drop_visual_p=drop_visual_p,
        )

        self.context = nn.GRU(
            input_size=d_model,
            hidden_size=context_hidden_dim,
            num_layers=context_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=context_dropout if context_num_layers > 1 else 0.0,
        )

        context_dim = context_hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(context_dim, classifier_hidden_dim),  # BiGRU 输出 -> 分类隐藏层
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        mask: torch.Tensor,
        return_gate: bool = False,
    ):
        batch_size, max_len, _ = text.shape

        flat_text = text.reshape(batch_size * max_len, -1)  # [B*L, text_dim]
        flat_audio = audio.reshape(batch_size * max_len, -1)
        flat_visual = visual.reshape(batch_size * max_len, -1)

        flat_fused, flat_gate = self.fusion.fuse(flat_text, flat_audio, flat_visual)
        fused = flat_fused.reshape(batch_size, max_len, -1)  # [B, L, d_model]

        lengths = mask.sum(dim=1).cpu()  # 每个 dialogue 的真实 utterance 数
        packed = nn.utils.rnn.pack_padded_sequence(
            fused,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.context(packed)
        context_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_len,
        )

        logits = self.classifier(context_output)  # [B, L, num_classes]
        if return_gate:
            gate = flat_gate.reshape(batch_size, max_len, 3)
            return logits, gate
        return logits


def build_dgf_context_model(config: dict | None = None) -> DGFContextClassifier:
    """按 YAML 配置构造 DGF + BiGRU context 模型。"""
    config = config or {}
    return DGFContextClassifier(
        text_dim=int(config.get("output_dim_text", 768)),
        audio_dim=int(config.get("output_dim_audio", 768)),
        visual_dim=int(config.get("output_dim_visual", 512)),
        d_model=int(config.get("d_model", 256)),
        projection_dropout=float(config.get("projection_dropout", 0.3)),
        use_layernorm=bool(config.get("use_layernorm", True)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 128)),
        gate_dropout=float(config.get("gate_dropout", 0.2)),
        num_classes=int(config.get("num_classes", 7)),
        use_modality_dropout=bool(config.get("use_modality_dropout", True)),
        modality_dropout_p=float(config.get("modality_dropout_p", 0.3)),
        drop_text_p=float(config.get("drop_text_p", 0.1)),
        drop_audio_p=float(config.get("drop_audio_p", 0.2)),
        drop_visual_p=float(config.get("drop_visual_p", 0.2)),
        context_hidden_dim=int(config.get("context_hidden_dim", 256)),
        context_num_layers=int(config.get("context_num_layers", 1)),
        bidirectional=bool(config.get("bidirectional", True)),
        context_dropout=float(config.get("context_dropout", 0.3)),
        classifier_hidden_dim=int(config.get("classifier_hidden_dim", 256)),
        classifier_dropout=float(config.get("classifier_dropout", 0.3)),
    )
