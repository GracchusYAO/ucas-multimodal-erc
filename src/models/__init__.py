"""Model components for multimodal ERC."""

from src.models.baselines import (
    AudioOnlyClassifier,
    AudioHubertDeepClassifier,
    AudioHubertOnlyClassifier,
    AudioHubertProsodyOnlyClassifier,
    AudioHubertProsodyDeepClassifier,
    AudioHubertStatsDeepClassifier,
    AudioHubertStatsOnlyClassifier,
    AudioProsodyOnlyClassifier,
    ConcatClassifier,
    ConcatTAVClassifier,
    TextOnlyClassifier,
    TextAudioClassifier,
    TextAudioHubertClassifier,
    TextAudioHubertProsodyClassifier,
    TextAudioHubertStatsClassifier,
    TextVisualFaceClassifier,
    TextVisualClassifier,
    VisualFaceOnlyClassifier,
    VisualOnlyClassifier,
    build_audio_only_model,
    build_audio_hubert_mlp_model,
    build_audio_hubert_only_model,
    build_audio_hubert_prosody_mlp_model,
    build_audio_hubert_prosody_only_model,
    build_audio_hubert_stats_mlp_model,
    build_audio_hubert_stats_only_model,
    build_audio_prosody_only_model,
    build_baseline_model,
    build_concat_tav_model,
    build_text_audio_model,
    build_text_audio_hubert_model,
    build_text_audio_hubert_prosody_model,
    build_text_audio_hubert_stats_model,
    build_text_only_model,
    build_text_visual_face_model,
    build_text_visual_model,
    build_visual_face_only_model,
    build_visual_only_model,
)
from src.models.context import DGFContextClassifier, build_dgf_context_model
from src.models.fusion import (
    AsymmetricQualityLogitFusionClassifier,
    DynamicGatedFusionClassifier,
    LateFusionHubertClassifier,
    QualityLateFusionHubertClassifier,
    build_asymmetric_quality_logit_fusion_model,
    build_dgf_model,
    build_late_fusion_hubert_model,
    build_quality_late_fusion_hubert_model,
)


def build_model(config: dict | None = None):
    """根据 model_name 构造当前实验模型。"""
    config = config or {}
    model_name = config.get("model_name", "text_only")
    if model_name in {
        "text_only",
        "audio_only",
        "audio_hubert_only",
        "audio_hubert_stats_only",
        "audio_prosody_only",
        "audio_hubert_prosody_only",
        "audio_hubert_mlp",
        "audio_hubert_stats_mlp",
        "audio_hubert_prosody_mlp",
        "visual_only",
        "text_audio",
        "text_audio_hubert",
        "text_audio_hubert_stats",
        "text_audio_hubert_prosody",
        "text_visual",
        "text_visual_face",
        "concat_tav",
        "concat_tav_hubert_stats_face",
        "visual_face_only",
    }:
        return build_baseline_model(config)
    if model_name in {"dgf", "dgf_dropout"}:
        return build_dgf_model(config)
    if model_name == "dgf_context":
        return build_dgf_context_model(config)
    if model_name in {"late_fusion_hubert", "late_fusion_hubert_face", "late_fusion_hubert_stats"}:
        return build_late_fusion_hubert_model(config)
    if model_name == "quality_late_fusion_hubert":
        return build_quality_late_fusion_hubert_model(config)
    if model_name == "asym_quality_logit_fusion":
        return build_asymmetric_quality_logit_fusion_model(config)
    raise ValueError(f"Unsupported model_name: {model_name}")

__all__ = [
    "AsymmetricQualityLogitFusionClassifier",
    "ConcatTAVClassifier",
    "ConcatClassifier",
    "DGFContextClassifier",
    "DynamicGatedFusionClassifier",
    "LateFusionHubertClassifier",
    "QualityLateFusionHubertClassifier",
    "TextOnlyClassifier",
    "TextAudioClassifier",
    "TextAudioHubertClassifier",
    "TextAudioHubertStatsClassifier",
    "TextAudioHubertProsodyClassifier",
    "TextVisualClassifier",
    "TextVisualFaceClassifier",
    "AudioOnlyClassifier",
    "AudioHubertDeepClassifier",
    "AudioHubertOnlyClassifier",
    "AudioHubertStatsDeepClassifier",
    "AudioHubertStatsOnlyClassifier",
    "AudioProsodyOnlyClassifier",
    "AudioHubertProsodyDeepClassifier",
    "AudioHubertProsodyOnlyClassifier",
    "VisualOnlyClassifier",
    "VisualFaceOnlyClassifier",
    "build_audio_only_model",
    "build_audio_hubert_mlp_model",
    "build_audio_hubert_only_model",
    "build_audio_hubert_stats_mlp_model",
    "build_audio_hubert_stats_only_model",
    "build_audio_hubert_prosody_mlp_model",
    "build_audio_prosody_only_model",
    "build_audio_hubert_prosody_only_model",
    "build_baseline_model",
    "build_asymmetric_quality_logit_fusion_model",
    "build_concat_tav_model",
    "build_text_audio_model",
    "build_text_audio_hubert_model",
    "build_text_audio_hubert_stats_model",
    "build_text_audio_hubert_prosody_model",
    "build_dgf_context_model",
    "build_dgf_model",
    "build_late_fusion_hubert_model",
    "build_quality_late_fusion_hubert_model",
    "build_model",
    "build_text_only_model",
    "build_text_visual_face_model",
    "build_text_visual_model",
    "build_visual_face_only_model",
    "build_visual_only_model",
]
