# Single Modality Improvement Plan

Date: 2026-05-11

## Why the Current Gate Gain Is Small

Current clean frozen-feature results:

| Model | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|
| `text_only` | 0.5722 | 0.4158 |
| `audio_hubert_only` | 0.3185 | 0.2182 |
| `audio_hubert_stats_only` | 0.3350 | 0.2161 |
| `audio_hubert_prosody_only` | 0.3406 | 0.2398 |
| `visual_only` | 0.2724 | 0.1807 |
| `visual_face_only` | 0.2888 | 0.1636 |
| `concat_tav` | 0.5794 | 0.4067 |
| `quality_late_fusion_hubert_d512` | 0.5981 | 0.4390 |

The gate is not the main bottleneck right now. The main bottleneck is that audio and visual branches are much weaker than text, so the fusion model can only use them cautiously. The current improvement over text-only is real:

```text
0.5722 -> 0.5981  (+0.0259 weighted F1)
```

But it is not large enough to make a strong project story unless we improve at least one non-text modality.

## Per-class Observation

Compared with `text_only`, the current main model improves:

- anger: `0.4308 -> 0.4533`
- fear: `0.1692 -> 0.2105`
- joy: `0.5374 -> 0.5637`
- neutral: `0.7058 -> 0.7372`
- sadness: `0.3242 -> 0.3663`

It slightly hurts surprise:

- surprise: `0.5440 -> 0.5398`

This suggests the gate is helping several categories a little, but audio/visual are not strong enough to create a decisive correction signal.

## Clean Improvement Targets

Do not use offline gate or fine-tuned text as the main solution. Improve the branch inputs and simple branch classifiers first.

### Priority 1: Audio Branch

Keep HuBERT as the main audio feature and treat prosody as an auxiliary candidate.

Planned experiments:

1. Stronger audio-only classifier:
   - replace shallow `Linear -> ReLU -> Dropout -> Linear` with a small 2-layer MLP for audio-only;
   - use stronger dropout and weight decay;
   - compare on `audio_hubert`, `audio_hubert_stats`, `audio_hubert_prosody`.
2. If the audio-only F1 improves, only then plug the better audio representation into `quality_late_fusion_hubert_d512`.
3. Keep the final model simple: one improved audio feature, not multiple audio experts.

Success criterion:

```text
audio-only weighted F1 >= 0.36
main gated model weighted F1 > 0.60
```

### Priority 2: Visual Branch

Current CLIP visual features are weak because they mostly capture scene/semantic cues, not facial expression.

Planned experiments:

1. Keep `visual_clip` as the stable baseline.
2. Treat `visual_face_clip` as exploratory only; it is face-centered CLIP, not an expression model.
3. If we add one new model, it should be a real facial-expression feature extractor, not another generic image encoder.

Success criterion:

```text
visual-only weighted F1 >= 0.33
text+visual or gated model improves over text_only more than current visual branch
```

### Priority 3: Text Branch

Text is already strong, but a cleaner frozen-feature improvement is acceptable if it is used fairly in both text-only and multimodal models.

Candidate:

- context-aware frozen text feature: encode previous utterances plus current utterance, without fine-tuning.

Important rule:

If we improve text features, the comparison must be:

```text
context_text_only
vs
context_text + audio + visual gated fusion
```

Otherwise the reported gate gain would be unfair.

## What Not to Do

Avoid these as main results:

- more offline gate variants;
- more text fine-tune ensembles;
- adding many auxiliary experts;
- reporting a high result that cannot be explained as one coherent model.

These can stay in work logs, but not in the main report table.

## Next Concrete Step

Run a focused audio improvement round:

1. Add a stronger but still simple audio MLP classifier.
2. Train:
   - `audio_hubert_stats_only`
   - `audio_hubert_prosody_only`
3. If either improves clearly, train exactly one gated model with that audio feature.
4. Compare against:
   - `text_only`
   - `concat_tav`
   - `quality_late_fusion_hubert_d512`

This keeps the project clean while directly addressing the small gate-vs-text-only margin.

## Audio MLP Check Result

Implemented a stronger two-layer single-modality MLP:

```text
DeepSingleModalityClassifier
Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm -> GELU -> Dropout -> Linear
```

Configs:

```text
configs/audio_hubert_mlp.yaml
configs/audio_hubert_stats_mlp.yaml
configs/audio_hubert_prosody_mlp.yaml
```

Results:

| Model | Test Weighted F1 | Test Macro F1 | Conclusion |
|---|---:|---:|---|
| `audio_hubert_only` | 0.3185 | 0.2182 | shallow baseline |
| `audio_hubert_mlp` | 0.3240 | 0.2205 | tiny gain only |
| `audio_hubert_stats_only` | 0.3350 | 0.2161 | shallow baseline |
| `audio_hubert_stats_mlp` | 0.2299 | 0.1812 | worse |
| `audio_hubert_prosody_only` | 0.3406 | 0.2398 | shallow baseline |
| `audio_hubert_prosody_mlp` | 0.3321 | 0.2238 | worse |

Conclusion:

The audio bottleneck is not mainly caused by the shallow classifier head. Deeper MLPs overfit or destabilize the weak audio signal. For the main project, keep the simpler audio branch unless a genuinely better audio representation is introduced.

Updated priority:

1. Do not merge the deep audio MLP into the main gated model.
2. Keep `audio_hubert_prosody_only` as an exploratory note, not a main result.
3. Next real opportunity is visual expression features or context-aware frozen text, not another audio head.
