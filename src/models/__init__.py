"""Model components for multimodal ERC."""

from src.models.baselines import (
    AudioOnlyClassifier,
    AudioHubertOnlyClassifier,
    ConcatClassifier,
    ConcatTAVClassifier,
    TextOnlyClassifier,
    TextAudioClassifier,
    TextAudioHubertClassifier,
    TextVisualClassifier,
    VisualOnlyClassifier,
    build_audio_only_model,
    build_audio_hubert_only_model,
    build_baseline_model,
    build_concat_tav_model,
    build_text_audio_model,
    build_text_audio_hubert_model,
    build_text_only_model,
    build_text_visual_model,
    build_visual_only_model,
)
from src.models.context import DGFContextClassifier, build_dgf_context_model
from src.models.fusion import (
    DynamicGatedFusionClassifier,
    LateFusionHubertClassifier,
    build_dgf_model,
    build_late_fusion_hubert_model,
)


def build_model(config: dict | None = None):
    """根据 model_name 构造当前实验模型。"""
    config = config or {}
    model_name = config.get("model_name", "text_only")
    if model_name in {
        "text_only",
        "audio_only",
        "audio_hubert_only",
        "visual_only",
        "text_audio",
        "text_audio_hubert",
        "text_visual",
        "concat_tav",
    }:
        return build_baseline_model(config)
    if model_name in {"dgf", "dgf_dropout"}:
        return build_dgf_model(config)
    if model_name == "dgf_context":
        return build_dgf_context_model(config)
    if model_name == "late_fusion_hubert":
        return build_late_fusion_hubert_model(config)
    raise ValueError(f"Unsupported model_name: {model_name}")

__all__ = [
    "ConcatTAVClassifier",
    "ConcatClassifier",
    "DGFContextClassifier",
    "DynamicGatedFusionClassifier",
    "LateFusionHubertClassifier",
    "TextOnlyClassifier",
    "TextAudioClassifier",
    "TextAudioHubertClassifier",
    "TextVisualClassifier",
    "AudioOnlyClassifier",
    "AudioHubertOnlyClassifier",
    "VisualOnlyClassifier",
    "build_audio_only_model",
    "build_audio_hubert_only_model",
    "build_baseline_model",
    "build_concat_tav_model",
    "build_text_audio_model",
    "build_text_audio_hubert_model",
    "build_dgf_context_model",
    "build_dgf_model",
    "build_late_fusion_hubert_model",
    "build_model",
    "build_text_only_model",
    "build_text_visual_model",
    "build_visual_only_model",
]
