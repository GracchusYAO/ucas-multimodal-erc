# Final Submission Guide

这份清单用于整理最终交给老师的代码范围。当前项目里保留了不少探索实验、失败分支和本地训练产物；最终提交时建议只提交下面这些能复现主线结果和支撑报告的文件。

## 1. 最终主线

最终建议提交并汇报的模型是：

```text
configs/context_lstm_residual_gated_fusion_fer_topk_d512.yaml
```

对应结果：

```text
Test Accuracy    = 0.6015
Test Weighted F1 = 0.6064
Test Macro F1    = 0.4401
```

核心结论：

```text
RoBERTa text + HuBERT audio + top-k FER visual
通过 dialogue BiLSTM 和 residual gated fusion 融合。
相比 text-only 和 simple concat 都有提升，并且 zero-modality 消融能说明音频和视觉有效。
```

## 2. 建议提交的代码文件

项目入口与数据：

```text
README.md
requirements.txt
.gitignore
src/__init__.py
src/dataset.py
src/feature_dataset.py
src/torch_import_patch.py
```

特征提取：

```text
src/extract_text_features.py
src/extract_audio_features.py
src/extract_audio_hubert_features.py
src/extract_visual_expression_features.py
```

其中 `extract_audio_features.py` 里有 mp4 音频解码 helper，`extract_audio_hubert_features.py` 会复用它，所以即使最终音频特征是 HuBERT，也建议保留这个文件。

训练、评估与可视化：

```text
src/train.py
src/evaluate.py
src/visualize.py
scripts/run_final_gated_multimodal.sh
```

模型代码：

```text
src/models/__init__.py
src/models/baselines.py
src/models/context.py
src/models/fusion.py
```

虽然最终主线主要使用 `context.py` 里的 `ContextLSTMResidualGatedFusionClassifier`，但 `context.py` 和 `__init__.py` 仍然依赖 `fusion.py` 中的部分基础模块，所以不要删除 `fusion.py`。

## 3. 建议提交的配置文件

主线配置：

```text
configs/context_lstm_residual_gated_fusion_fer_topk_d512.yaml
```

对照实验配置：

```text
configs/text_only.yaml
configs/audio_hubert_only.yaml
configs/visual_expression_topk_only.yaml
configs/concat_tav_hubert_expression_topk_d512.yaml
```

这几份配置足够支撑报告中的核心对比：

```text
text-only
audio-only
visual-only
simple concat
final residual gated fusion
```

## 4. 建议提交的说明文档

```text
experiment_discussion.md
work_log.md
multimodal_erc_todolist.md
```

其中 `experiment_discussion.md` 是最适合给同学或老师看的最终实验说明；`work_log.md` 记录了关键过程和失败尝试，答辩时如果被问到为什么不选其他方案，可以用它回溯。

## 5. 不建议提交的大文件

这些目录是本地生成产物或数据，不建议放进代码提交：

```text
data/meld/train_splits/
data/meld/dev_splits_complete/
data/meld/output_repeated_splits_test/
features/
checkpoints/
results/
logs/
.torch_cache/
```

原因：

- `data/meld/*_splits*` 是 MELD 原始视频，体积大，不适合进 Git。
- `features/` 是缓存特征，可以由提取脚本重新生成。
- `checkpoints/` 是模型权重，通常不随代码提交，除非老师明确要求提交最佳权重。
- `results/` 和 `logs/` 是实验输出，可以作为本地证据，但不适合塞进最终代码仓库。

如果老师要求提交可直接复现实验的压缩包，可以额外单独附：

```text
checkpoints/context_lstm_residual_gated_fusion_fer_topk_d512/best_context_lstm_residual_gated_fusion.pt
features/text_roberta/
features/audio_hubert/
features/visual_expression_topk/
```

这属于“实验附件”，不建议和代码混在一个 Git 提交里。

## 6. 可以不放进最终提交的旁支代码

下面这些文件或配置对应探索分支，不作为最终主线。它们可以留在研究仓库中，但如果要交一个干净版代码包，可以不包含。

旁支评估/ensemble：

```text
src/evaluate_logits_ensemble.py
src/evaluate_mixed_ensemble.py
src/evaluate_offline_gated_fusion.py
src/evaluate_text_ensemble.py
src/export_logits.py
```

文本 fine-tuning 分支：

```text
src/train_text_finetune.py
configs/text_finetune*.yaml
```

非最终音频尝试：

```text
src/combine_audio_features.py
src/extract_audio_emotion_features.py
src/extract_audio_prosody_features.py
configs/audio_emotion_only.yaml
configs/audio_hubert_stats*.yaml
configs/audio_hubert_prosody*.yaml
configs/audio_prosody_only.yaml
configs/context_lstm_residual_gated_fusion_fer_topk_hubert_stats_d512.yaml
configs/context_lstm_residual_gated_fusion_fer_topk_hubert_prosody_d512.yaml
configs/context_lstm_residual_gated_fusion_fer_topk_audio_emotion_d512.yaml
```

非最终视觉尝试：

```text
src/extract_visual_face_features.py
src/extract_visual_features.py
configs/visual_expression_affectnet_only.yaml
configs/visual_expression_compact_only.yaml
configs/text_visual*.yaml
configs/context_lstm_residual_gated_fusion_affectnet_d512.yaml
```

旧融合结构/容量探索：

```text
configs/dgf*.yaml
configs/late_fusion*.yaml
configs/quality_late_fusion*.yaml
configs/asym_quality_logit*.yaml
configs/context_residual_gated_fusion_fer.yaml
configs/context_lstm_residual_gated_fusion_fer.yaml
configs/context_lstm_residual_gated_fusion_fer_d512.yaml
configs/context_lstm_residual_gated_fusion_fer_d768.yaml
```

注意：这里说的是“最终提交包可以不包含”，不是立刻从研究仓库删除。它们对复盘实验路线仍然有用。

## 7. 是否建议现在删除失败模型代码

不建议马上从当前研究仓库里直接删除。

更稳妥的做法是：

1. 当前仓库保留完整实验历史，避免报告或答辩时找不到失败对照。
2. 用本文件作为最终提交清单，只把主线必要文件复制/打包给老师。
3. 如果后续确定要做一个干净仓库，再按第 6 节的旁支清单删除或移动到 `archive/`。

一句话：最终提交应该“少而清楚”，研究仓库可以“全而可追溯”。
