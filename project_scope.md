# Project Scope and Cleanup Notes

Date: 2026-05-11

这个文件用于把当前项目收束成一条清楚主线，避免后续继续把 fine-tune、offline ensemble、各种旁支结构混在一起。

## Main Story

课程项目主线应该是：

```text
MELD 多模态情绪识别
-> 冻结预训练特征
-> 单模态 baseline
-> 简单拼接 baseline
-> 我们的 quality-aware gated fusion
-> 消融实验说明 gate / quality / modality dropout 的价值
```

主结果不应该依赖 fine-tuned text ensemble 或 offline gate。那些可以作为补充讨论，但不能当作模型设计的核心贡献。

## Core Files

这些是后续真正要围绕它们写报告、跑实验和维护的文件。

### Dataset and Cached Features

```text
src/dataset.py
src/feature_dataset.py
```

用途：

- 读取 MELD CSV 和媒体路径；
- 读取已经缓存好的 text/audio/visual 特征；
- 检查不同模态样本顺序是否对齐。

### Feature Extraction

```text
src/extract_text_features.py
src/extract_audio_hubert_features.py
src/extract_visual_features.py
src/extract_visual_face_features.py
src/extract_audio_prosody_features.py
src/combine_audio_features.py
```

主线建议：

- text: `features/text_roberta`
- audio: 优先用 `features/audio_hubert`
- visual: 优先用 `features/visual_clip`

可继续探索：

- `audio_prosody` 和 `audio_hubert_prosody` 用于提升音频分支，但目前还没有稳定提升主模型；
- `visual_face_clip` 是 face-centered CLIP，不是真正 expression feature，可作为视觉增强候选，但不能夸大。

相对旁支：

```text
src/extract_audio_features.py
```

这是早期 Wav2Vec2 音频特征脚本。HuBERT 效果更好后，它只作为历史 baseline 保留。

### Training and Evaluation

```text
src/train.py
src/evaluate.py
src/visualize.py
```

这是主线训练、测试、画图入口。

主线训练命令应该尽量统一为：

```bash
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.train \
  --config configs/<config>.yaml \
  --device cuda \
  --output-dir results/<run_name> \
  --checkpoint-dir checkpoints/<run_name>
```

主线评估命令应该尽量统一为：

```bash
/home/gracchus/miniconda3/envs/workspace/bin/python -m src.evaluate \
  --config configs/<config>.yaml \
  --checkpoint checkpoints/<run_name>/best_<model_name>.pt \
  --split test \
  --device cuda \
  --output-dir results/evaluate/<run_name>_test
```

### Models

```text
src/models/baselines.py
src/models/fusion.py
src/models/context.py
src/models/__init__.py
```

主线模型：

- `TextOnlyClassifier`
- `AudioHubertOnlyClassifier`
- `VisualOnlyClassifier`
- `ConcatTAVClassifier`
- `DynamicGatedFusionClassifier`
- `LateFusionHubertClassifier`
- `QualityLateFusionHubertClassifier`

旁支/实验模型：

- `AsymmetricQualityLogitFusionClassifier`
- `AudioProsodyOnlyClassifier`
- `AudioHubertProsodyOnlyClassifier`
- `VisualFaceOnlyClassifier`
- context 版本暂时只作为补充，因为当前结果没有成为主结果。

## Core Configs

这些配置建议作为报告表格的主要来源。

```text
configs/text_only.yaml
configs/audio_hubert_only.yaml
configs/visual_only.yaml
configs/concat_tav.yaml
configs/dgf_dropout.yaml
configs/late_fusion_hubert.yaml
configs/quality_late_fusion_hubert_d512.yaml
```

其中 `quality_late_fusion_hubert_d512` 暂时是主模型候选：

```text
test weighted_f1 = 0.5981
test macro_f1    = 0.4390
```

## Useful Ablation Configs

这些不一定进主表，但适合放进消融表或讨论。

```text
configs/dgf_dropout_d128.yaml
configs/dgf_dropout_d512.yaml
configs/late_fusion_hubert_d128.yaml
configs/late_fusion_hubert_d512.yaml
configs/quality_late_fusion_hubert_d128.yaml
configs/quality_late_fusion_hubert_d512_aux02.yaml
configs/quality_late_fusion_hubert_d512_aux05.yaml
configs/text_audio_hubert.yaml
configs/text_visual.yaml
configs/text_visual_face.yaml
configs/audio_hubert_stats_only.yaml
configs/concat_tav_hubert_stats_face.yaml
```

## Side Branches

这些只适合放入附录或工作日志，不建议作为课程项目主线。

### Fine-tuned Text

```text
src/train_text_finetune.py
src/evaluate_text_ensemble.py
configs/text_finetune*.yaml
```

定位：

- 作为强文本上界；
- 说明 MELD 任务中文本模态非常强；
- 不作为我们多模态结构的主贡献。

### Offline Fusion / Logits Ensemble

```text
src/export_logits.py
src/evaluate_mixed_ensemble.py
src/evaluate_offline_gated_fusion.py
src/evaluate_logits_ensemble.py
scripts/run_final_gated_multimodal.sh
```

定位：

- 这是 post-hoc ensemble / 后处理融合；
- 可以用来分析多模态预测是否有互补性；
- 不作为端到端模型主结果。

### Exploratory Architecture Branches

```text
configs/asym_quality_logit_*.yaml
configs/quality_late_fusion_hubert_asym*.yaml
configs/quality_late_fusion_hubert_prosody*.yaml
configs/audio_prosody_only.yaml
configs/audio_hubert_prosody_only.yaml
configs/concat_tav_hubert_prosody.yaml
```

定位：

- 这些是为了确认“扩大音频/视觉容量、加入 prosody 是否有帮助”；
- 目前结论是音频分支有改善，但没有稳定提升主模型；
- 后续若继续提升单模态，可以从这里挑少量有效内容并入主线。

## Local Artifacts

这些目录不应该提交到 Git：

```text
data/
features/
results/
checkpoints/
logs/
```

当前 `.gitignore` 已经忽略：

```text
data/
features/
results/
checkpoints/
runs/
wandb/
```

建议补充忽略：

```text
logs/
*.pdf
```

如果 `experiment_discussion.md` 是临时讨论材料，也不建议进入最终提交；如果之后要提交，应改名成更正式的实验讨论文档。

## Current Main Results

主线 frozen-feature 实验目前最清楚的一组是：

| Model | Test Weighted F1 | Test Macro F1 | Role |
|---|---:|---:|---|
| `text_only` | 0.5722 | 0.4158 | strongest clean single-modality baseline |
| `concat_tav` | 0.5794 | 0.4067 | simple multimodal baseline |
| `dgf_dropout` | 0.5903 | 0.4263 | gated fusion baseline |
| `quality_late_fusion_hubert_d512` | 0.5981 | 0.4390 | current main model |

这个结果说明 gate 有用，但提升幅度还不够大。后续重点应该是让单模态分支更可靠，尤其是 audio / visual，而不是继续堆 offline gate 或 ensemble。

## Next Improvement Direction

短期目标：

```text
让主模型相对 text_only 的提升从约 +2.6 weighted F1 尽量扩大到 +3.5 到 +5.0。
```

优先级：

1. Text branch: 做一个更公平、更强的 frozen text baseline，例如 context-aware RoBERTa feature，而不是直接上 fine-tune。
2. Audio branch: 保留 HuBERT 主线，谨慎加入 prosody；只有当 audio-only 和 fusion 同时提升时才并入主模型。
3. Visual branch: 当前 CLIP 更偏语义，不是表情。下一步应该尝试真正 expression-oriented 的视觉特征，而不是继续扩大 CLIP 维度。
4. Fusion: 固定在 `quality_late_fusion_hubert_d512` 附近，只做必要小改，避免继续发散。
