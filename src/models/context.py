"""Dialogue context encoder components."""

from __future__ import annotations

import torch
from torch import nn

from src.models.fusion import DynamicGatedFusionClassifier


def _context_audio_dim(config: dict) -> int:
    """根据当前启用的音频模态，选择对应的输入维度。"""
    modalities = config.get("modalities", {})

    if modalities.get("audio_hubert_prosody", False):
        return int(config.get("output_dim_audio_hubert_prosody", config.get("output_dim_audio_hubert", 883)))
    if modalities.get("audio_emotion", False):
        return int(config.get("output_dim_audio_emotion", 782))
    if modalities.get("audio_prosody", False):
        return int(config.get("output_dim_audio_prosody", 115))
    if modalities.get("audio_hubert_stats", False):
        return int(config.get("output_dim_audio_hubert_stats", 1536))
    if modalities.get("audio_hubert", False):
        return int(config.get("output_dim_audio_hubert", 768))
    return int(config.get("output_dim_audio", 768))


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


class ContextResidualGatedFusionClassifier(nn.Module):
    """Context residual fusion：文本做主判断，音频/视觉做门控修正。"""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 782,
        d_model: int = 256,
        projection_dropout: float = 0.3,
        use_layernorm: bool = True,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
        text_gate_bias: float = 1.5,
        residual_scale: float = 1.0,
        num_classes: int = 7,
        context_hidden_dim: int = 256,
        context_num_layers: int = 1,
        bidirectional: bool = True,
        context_dropout: float = 0.3,
        branch_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.residual_scale = residual_scale

        # 三个模态先各自投影，再一起送进 dialogue context。
        self.text_proj = self._make_projection(text_dim, d_model, projection_dropout, use_layernorm)
        self.audio_proj = self._make_projection(audio_dim, d_model, projection_dropout, use_layernorm)
        self.visual_proj = self._make_projection(visual_dim, d_model, projection_dropout, use_layernorm)

        self.context = nn.GRU(
            input_size=d_model * 3,  # context 同时看 text/audio/visual
            hidden_size=context_hidden_dim,
            num_layers=context_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=context_dropout if context_num_layers > 1 else 0.0,
        )

        context_dim = context_hidden_dim * (2 if bidirectional else 1)
        branch_input_dim = d_model + context_dim

        self.text_head = self._make_head(branch_input_dim, num_classes, branch_dropout)
        self.audio_delta_head = self._make_head(branch_input_dim, num_classes, branch_dropout)
        self.visual_delta_head = self._make_head(branch_input_dim, num_classes, branch_dropout)

        gate_input_dim = d_model * 3 + context_dim + num_classes * 3
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),  # gate 同时看表示、上下文和三路 logits/delta
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        with torch.no_grad():
            # 初始偏向保留文本基座，后续训练再学习是否让音频/视觉修正。
            self.gate[-1].bias.copy_(torch.tensor([text_gate_bias, 0.0, 0.0]))

    @staticmethod
    def _make_projection(
        input_dim: int,
        output_dim: int,
        dropout: float,
        use_layernorm: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if use_layernorm:
            layers.append(nn.LayerNorm(output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_head(input_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # 分支表示 + dialogue context -> logits/delta
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def encode_context(
        self,
        z_text: torch.Tensor,
        z_audio: torch.Tensor,
        z_visual: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """按 dialogue 顺序建模上下文，只让真实 utterance 参与 GRU。"""
        context_input = torch.cat([z_text, z_audio, z_visual], dim=-1)
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            context_input,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.context(packed)
        context_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=context_input.size(1),
        )
        return context_output

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        mask: torch.Tensor,
        return_gate: bool = False,
    ):
        batch_size, max_len, _ = text.shape

        flat_text = text.reshape(batch_size * max_len, -1)
        flat_audio = audio.reshape(batch_size * max_len, -1)
        flat_visual = visual.reshape(batch_size * max_len, -1)

        z_text = self.text_proj(flat_text).reshape(batch_size, max_len, -1)
        z_audio = self.audio_proj(flat_audio).reshape(batch_size, max_len, -1)
        z_visual = self.visual_proj(flat_visual).reshape(batch_size, max_len, -1)
        z_context = self.encode_context(z_text, z_audio, z_visual, mask)

        text_input = torch.cat([z_text, z_context], dim=-1)
        audio_input = torch.cat([z_audio, z_context], dim=-1)
        visual_input = torch.cat([z_visual, z_context], dim=-1)

        text_logits = self.text_head(text_input)  # 文本基座：主判断
        audio_delta = self.audio_delta_head(audio_input)  # 音频只学习如何修正文本 logits
        visual_delta = self.visual_delta_head(visual_input)  # FER 视觉同样只做修正

        gate_input = torch.cat(
            [
                z_text,
                z_audio,
                z_visual,
                z_context,
                text_logits,
                audio_delta,
                visual_delta,
            ],
            dim=-1,
        )
        gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
        logits = text_logits + self.residual_scale * (
            gate_weights[..., 1:2] * audio_delta
            + gate_weights[..., 2:3] * visual_delta
        )

        if return_gate:
            return logits, gate_weights
        return logits


class ContextLSTMResidualGatedFusionClassifier(nn.Module):
    """LSTM residual fusion：文本做基座，音频/视觉用独立 sigmoid gate 做修正。"""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 782,
        d_model: int = 256,
        projection_dropout: float = 0.3,
        use_layernorm: bool = True,
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.2,
        audio_gate_bias: float = -1.0,
        visual_gate_bias: float = -2.0,
        audio_residual_scale: float = 1.0,
        visual_residual_scale: float = 0.4,
        num_classes: int = 7,
        context_hidden_dim: int = 256,
        context_num_layers: int = 1,
        bidirectional: bool = True,
        context_dropout: float = 0.3,
        branch_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_residual_scale = audio_residual_scale
        self.visual_residual_scale = visual_residual_scale

        # 三个模态仍然先压到同一个 d_model，方便 LSTM 在同一空间里看上下文。
        self.text_proj = ContextResidualGatedFusionClassifier._make_projection(
            text_dim,
            d_model,
            projection_dropout,
            use_layernorm,
        )
        self.audio_proj = ContextResidualGatedFusionClassifier._make_projection(
            audio_dim,
            d_model,
            projection_dropout,
            use_layernorm,
        )
        self.visual_proj = ContextResidualGatedFusionClassifier._make_projection(
            visual_dim,
            d_model,
            projection_dropout,
            use_layernorm,
        )

        self.context = nn.LSTM(
            input_size=d_model * 3,  # LSTM 同时读三路证据，hidden state 作为 dialogue memory
            hidden_size=context_hidden_dim,
            num_layers=context_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=context_dropout if context_num_layers > 1 else 0.0,
        )

        context_dim = context_hidden_dim * (2 if bidirectional else 1)
        branch_input_dim = d_model + context_dim

        self.text_head = ContextResidualGatedFusionClassifier._make_head(
            branch_input_dim,
            num_classes,
            branch_dropout,
        )
        self.audio_delta_head = ContextResidualGatedFusionClassifier._make_head(
            branch_input_dim,
            num_classes,
            branch_dropout,
        )
        self.visual_delta_head = ContextResidualGatedFusionClassifier._make_head(
            branch_input_dim,
            num_classes,
            branch_dropout,
        )

        gate_input_dim = d_model * 3 + context_dim + num_classes * 3
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),  # gate 看当前 utterance、上下文和三路预测
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 2),  # 只预测 audio / visual 两个修正门，文本基座始终保留
        )
        with torch.no_grad():
            # 初始化时让模型谨慎使用弱模态，避免视觉/音频一上来把文本判断拉偏。
            self.gate[-1].bias.copy_(torch.tensor([audio_gate_bias, visual_gate_bias]))

    def encode_context(
        self,
        z_text: torch.Tensor,
        z_audio: torch.Tensor,
        z_visual: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """按 dialogue 顺序跑 LSTM，只让真实 utterance 进入 recurrent context。"""
        context_input = torch.cat([z_text, z_audio, z_visual], dim=-1)
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            context_input,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.context(packed)
        context_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=context_input.size(1),
        )
        return context_output

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        mask: torch.Tensor,
        return_gate: bool = False,
    ):
        batch_size, max_len, _ = text.shape

        flat_text = text.reshape(batch_size * max_len, -1)
        flat_audio = audio.reshape(batch_size * max_len, -1)
        flat_visual = visual.reshape(batch_size * max_len, -1)

        z_text = self.text_proj(flat_text).reshape(batch_size, max_len, -1)
        z_audio = self.audio_proj(flat_audio).reshape(batch_size, max_len, -1)
        z_visual = self.visual_proj(flat_visual).reshape(batch_size, max_len, -1)
        z_context = self.encode_context(z_text, z_audio, z_visual, mask)

        text_input = torch.cat([z_text, z_context], dim=-1)
        audio_input = torch.cat([z_audio, z_context], dim=-1)
        visual_input = torch.cat([z_visual, z_context], dim=-1)

        text_logits = self.text_head(text_input)  # 主预测来自文本 + dialogue context
        audio_delta = self.audio_delta_head(audio_input)  # 音频学习“改多少”，不是单独抢最终分类权
        visual_delta = self.visual_delta_head(visual_input)  # FER 视觉同样学习 residual correction

        gate_input = torch.cat(
            [
                z_text,
                z_audio,
                z_visual,
                z_context,
                text_logits,
                audio_delta,
                visual_delta,
            ],
            dim=-1,
        )
        audio_gate, visual_gate = torch.sigmoid(self.gate(gate_input)).chunk(2, dim=-1)
        logits = (
            text_logits
            + audio_gate * self.audio_residual_scale * audio_delta
            + visual_gate * self.visual_residual_scale * visual_delta
        )

        if return_gate:
            # 这里 gate_text=1 表示文本基座始终保留；audio/visual 是额外修正强度，不再相加为 1。
            text_gate = torch.ones_like(audio_gate)
            gate_weights = torch.cat([text_gate, audio_gate, visual_gate], dim=-1)
            return logits, gate_weights
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


def build_context_residual_gated_fusion_model(
    config: dict | None = None,
) -> ContextResidualGatedFusionClassifier:
    """按 YAML 配置构造 context-aware residual gated fusion 模型。"""
    config = config or {}
    return ContextResidualGatedFusionClassifier(
        text_dim=int(config.get("output_dim_text", 768)),
        audio_dim=_context_audio_dim(config),
        visual_dim=int(config.get("output_dim_visual", 782)),
        d_model=int(config.get("d_model", 256)),
        projection_dropout=float(config.get("projection_dropout", 0.3)),
        use_layernorm=bool(config.get("use_layernorm", True)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 128)),
        gate_dropout=float(config.get("gate_dropout", 0.2)),
        text_gate_bias=float(config.get("text_gate_bias", 1.5)),
        residual_scale=float(config.get("residual_scale", 1.0)),
        num_classes=int(config.get("num_classes", 7)),
        context_hidden_dim=int(config.get("context_hidden_dim", 256)),
        context_num_layers=int(config.get("context_num_layers", 1)),
        bidirectional=bool(config.get("bidirectional", True)),
        context_dropout=float(config.get("context_dropout", 0.3)),
        branch_dropout=float(config.get("branch_dropout", 0.3)),
    )


def build_context_lstm_residual_gated_fusion_model(
    config: dict | None = None,
) -> ContextLSTMResidualGatedFusionClassifier:
    """按 YAML 配置构造 LSTM context-aware residual gated fusion 模型。"""
    config = config or {}
    return ContextLSTMResidualGatedFusionClassifier(
        text_dim=int(config.get("output_dim_text", 768)),
        audio_dim=_context_audio_dim(config),
        visual_dim=int(config.get("output_dim_visual", 782)),
        d_model=int(config.get("d_model", 256)),
        projection_dropout=float(config.get("projection_dropout", 0.3)),
        use_layernorm=bool(config.get("use_layernorm", True)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 128)),
        gate_dropout=float(config.get("gate_dropout", 0.2)),
        audio_gate_bias=float(config.get("audio_gate_bias", -1.0)),
        visual_gate_bias=float(config.get("visual_gate_bias", -2.0)),
        audio_residual_scale=float(config.get("audio_residual_scale", 1.0)),
        visual_residual_scale=float(config.get("visual_residual_scale", 0.4)),
        num_classes=int(config.get("num_classes", 7)),
        context_hidden_dim=int(config.get("context_hidden_dim", 256)),
        context_num_layers=int(config.get("context_num_layers", 1)),
        bidirectional=bool(config.get("bidirectional", True)),
        context_dropout=float(config.get("context_dropout", 0.3)),
        branch_dropout=float(config.get("branch_dropout", 0.3)),
    )
