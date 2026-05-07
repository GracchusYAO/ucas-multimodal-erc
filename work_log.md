# Work Log

## 2026-04-28: MELD Raw Data Preparation

### Summary

Downloaded and prepared the MELD raw dataset for the multimodal emotion recognition project. The data was organized under:

```text
data/meld/
  train_sent_emo.csv
  dev_sent_emo.csv
  test_sent_emo.csv
  train_splits/
  dev_splits_complete/
  output_repeated_splits_test/
```

The `data/` directory is intentionally ignored by Git, so the raw dataset will not be pushed to the remote repository.

### Data Source

- Official MELD project page: https://affective-meld.github.io/
- Official GitHub repository: https://github.com/declare-lab/MELD
- Downloaded raw package: `MELD.Raw.tar.gz`

### Validation Results

CSV row counts:

```text
train_sent_emo.csv: 9989 utterances
dev_sent_emo.csv:   1109 utterances
test_sent_emo.csv:  2610 utterances
```

Dialogue counts parsed from the CSV files:

```text
train: 1038 unique Dialogue_ID values
dev:    114 unique Dialogue_ID values
test:   280 unique Dialogue_ID values
```

The train metadata has a gap at `Dialogue_ID=60`. The parser preserves the official CSV ids instead of renumbering them, so dialogue batching remains aligned with the source metadata.

This gap does not mean that one existing dialogue failed to download. It means no row in `train_sent_emo.csv` uses `Dialogue_ID=60`. The most likely explanation is that MELD preserves dialogue ids after its dataset-cleaning/filtering process. The official README notes that some EmotionLines cases were filtered because they mixed multiple natural dialogues or failed timestamp/scene constraints, so ids should be treated as stable identifiers rather than a guaranteed contiguous sequence.

Extracted video counts:

```text
train_splits/:                 9989 mp4 files
dev_splits_complete/:          1112 mp4 files
output_repeated_splits_test/:  2747 mp4 files
```

Metadata-to-media check:

```text
train: rows=9989, missing_media=0
dev:   rows=1109, missing_media=1
test:  rows=2610, missing_media=0
```

### Noted Dataset Issue

One dev-set utterance has metadata in `dev_sent_emo.csv`, but the corresponding raw video file is absent from the official raw package.

Missing media:

```text
data/meld/dev_splits_complete/dia110_utt7.mp4
```

Corresponding metadata:

```text
Dialogue_ID: 110
Utterance_ID: 7
Speaker: Phoebe
Utterance: Did they tell you anything?
Emotion: neutral
Sentiment: neutral
Season: 6
Episode: 11
StartTime: 00:07:11,764
EndTime: 00:07:13,890
```

Local directory check shows that nearby files exist:

```text
dia110_utt0.mp4
dia110_utt1.mp4
dia110_utt2.mp4
dia110_utt3.mp4
dia110_utt4.mp4
dia110_utt5.mp4
dia110_utt6.mp4
dia110_utt8.mp4
dia110_utt9.mp4
```

This suggests a small issue in the official raw data package rather than a project-side download or extraction error. The downloaded archive passed `gzip -t`, and train/test metadata-to-media checks are complete.

### Project Impact

The issue affects 1 utterance out of 1109 in the dev split, about 0.09% of the dev metadata. It should have negligible impact on text-only experiments and a very small impact on audio/visual feature extraction or multimodal evaluation.

### Planned Handling

For implementation, keep the official split and handle missing media explicitly:

- Keep the dev metadata row so labels and dialogue order remain aligned with the official split.
- For text features, process the utterance normally.
- For audio and visual features, use a zero vector or missing-modality mask for `dia110_utt7.mp4`.
- Log missing media during feature extraction instead of failing silently.
- Make missing-modality handling compatible with the planned modality dropout and gated fusion design.

## 2026-04-28: Text Feature Extraction Script

Implemented `src/extract_text_features.py` for frozen RoBERTa utterance-level text embeddings.

Supported behavior:

- Load MELD utterances through `src.dataset`.
- Process one or more official splits.
- Read text settings from `configs/text_only.yaml` or CLI arguments.
- Support `mean` pooling and `cls` pooling.
- Save outputs to `features/text_roberta/{split}.pt`.
- Save features together with labels, dialogue ids, utterance ids, keys, texts, speakers, emotions, and sentiments.
- Provide `--dry-run` mode for checking split sizes and output paths without loading pretrained models.

Verified dry-run command:

```text
conda run -n workspace python -m src.extract_text_features --config configs/text_only.yaml --split dev --dry-run
```

Output:

```text
[dry-run] split=dev, utterances=1109, output=features/text_roberta/dev.pt
```

Environment note:

- `workspace` currently has `torch`, `torchaudio`, and `torchvision`.
- `workspace` does not currently have `transformers`, `tqdm`, or `pyyaml`.
- The script can still run `--dry-run` in `workspace` because YAML and progress-bar dependencies are optional/fallback for that mode.
- Real RoBERTa feature extraction will require `transformers` in the environment used to run the script.

## 2026-04-29: Code Simplification for Course Project Scope

Simplified the dataset and feature extraction scripts to better fit a course-project codebase rather than a production-style pipeline.

Main changes:

- Reduced `src/dataset.py` to the core MELD parsing functions and simple dialogue grouping.
- Removed extra dataset wrapper classes and overly defensive parsing options.
- Simplified text feature extraction while keeping RoBERTa mean/CLS pooling and `.pt` output metadata.
- Simplified audio feature extraction to use ffmpeg-based mp4 audio decoding directly, then Wav2Vec2 mean pooling.
- Simplified visual feature extraction to uniform frame sampling plus CLIP frame-feature averaging.
- Added more Chinese comments around dataset ids, missing media, pooling, zero-vector handling, and batching logic.

The four main scripts went from about 1523 lines to about 938 lines:

```text
src/dataset.py
src/extract_text_features.py
src/extract_audio_features.py
src/extract_visual_features.py
```

Verification after simplification:

```text
conda run -n workspace python -m compileall src/dataset.py src/extract_text_features.py src/extract_audio_features.py src/extract_visual_features.py
conda run -n workspace python -m src.dataset --data-root data/meld --split dev
conda run -n workspace python -m src.extract_text_features --config configs/text_only.yaml --split dev --dry-run
conda run -n workspace python -m src.extract_audio_features --split dev --dry-run
conda run -n workspace python -m src.extract_visual_features --split dev --dry-run
```

Also verified one MELD mp4 can be decoded by the simplified audio and visual helpers:

```text
audio waveform shape: (44715,)
visual frames: 4 frames, first frame shape (720, 1280, 3)
```

## 2026-04-30: Text-only Baseline Training Pipeline

Implemented the first training loop for the cached-feature setup.

Main changes:

- Added `src/feature_dataset.py` to load cached `.pt` features from `features/`.
- Added `TextOnlyClassifier` in `src/models/baselines.py`.
- Implemented `src/train.py` for the text-only baseline.
- The training script reads `configs/text_only.yaml`, uses AdamW, class-weighted cross entropy, dev weighted F1 for model selection, and early stopping.
- Training logs are written to CSV, and the best checkpoint is saved by dev weighted F1.
- All experiment config seeds were standardized to `114514`.

Smoke training command:

```text
conda run -n workspace python -u -m src.train --config configs/text_only.yaml --device cpu --max-epochs 20 --patience 5 --batch-size 64 --output-dir results/text_only_smoke --checkpoint-dir checkpoints/text_only_smoke 2>&1 | tee logs/train_text_only_smoke.log
```

Outputs:

```text
logs/train_text_only_smoke.log
results/text_only_smoke/train_log.csv
checkpoints/text_only_smoke/best_text_only.pt
```

Result:

```text
seed: 114514
best epoch: 19
dev loss: 1.5052
dev accuracy: 0.5077
dev weighted F1: 0.5299
dev macro F1: 0.3968
```

This confirms that cached RoBERTa text features can be loaded, batched, trained, evaluated, logged, and checkpointed end to end.

## 2026-04-30: Trimodal Concatenation Baseline

Implemented the second baseline model: Text + Audio + Visual feature concatenation.

Main changes:

- Added `ConcatTAVClassifier` in `src/models/baselines.py`.
- Generalized `src/train.py` so the same training loop supports both `text_only` and `concat_tav`.
- Updated `configs/concat_tav.yaml` with utterance-level batch size and seed `114514`.
- Verified cached text/audio/visual features are aligned through `src.feature_dataset`.

Smoke training command:

```text
conda run -n workspace python -u -m src.train --config configs/concat_tav.yaml --device cpu --max-epochs 20 --patience 5 --batch-size 64 --output-dir results/concat_tav_smoke --checkpoint-dir checkpoints/concat_tav_smoke 2>&1 | tee logs/train_concat_tav_smoke.log
```

Outputs:

```text
logs/train_concat_tav_smoke.log
results/concat_tav_smoke/train_log.csv
checkpoints/concat_tav_smoke/best_concat_tav.pt
```

Result:

```text
seed: 114514
best epoch: 20
dev loss: 1.4529
dev accuracy: 0.5239
dev weighted F1: 0.5464
dev macro F1: 0.4077
```

The concat baseline is slightly stronger than the text-only smoke result, so it is now ready to serve as the simple multimodal baseline before implementing gated fusion.

## 2026-04-30: Dynamic Reliability-Gated Fusion

Implemented the Dynamic Reliability-Gated Fusion model.

Main changes:

- Added `DynamicGatedFusionClassifier` in `src/models/fusion.py`.
- Projected text/audio/visual features into a shared `d_model` space.
- Added a gate MLP that outputs per-sample text/audio/visual weights.
- The model can return both fused features and gate weights for later visualization.
- Added modality dropout support in the same model class, controlled by `configs/dgf_dropout.yaml`.
- Updated `src/train.py` so `model_name: dgf` can use the same cached-feature training loop.

Smoke training command:

```text
conda run -n workspace python -u -m src.train --config configs/dgf.yaml --device cpu --max-epochs 20 --patience 5 --batch-size 64 --output-dir results/dgf_smoke --checkpoint-dir checkpoints/dgf_smoke 2>&1 | tee logs/train_dgf_smoke.log
```

Outputs:

```text
logs/train_dgf_smoke.log
results/dgf_smoke/train_log.csv
checkpoints/dgf_smoke/best_dgf.pt
```

Result:

```text
seed: 114514
best epoch: 13
dev loss: 1.5378
dev accuracy: 0.5573
dev weighted F1: 0.5695
dev macro F1: 0.4287
```

Current smoke comparison:

```text
text-only weighted F1: 0.5299
concat T+A+V weighted F1: 0.5464
DGF weighted F1: 0.5695
```

The gated fusion model is currently the strongest smoke result among the implemented models.

## 2026-04-30: DGF with Modality Dropout

Ran the Dynamic Reliability-Gated Fusion model with modality dropout enabled.

Smoke training command:

```text
conda run -n workspace python -u -m src.train --config configs/dgf_dropout.yaml --device cpu --max-epochs 20 --patience 5 --batch-size 64 --output-dir results/dgf_dropout_smoke --checkpoint-dir checkpoints/dgf_dropout_smoke 2>&1 | tee logs/train_dgf_dropout_smoke.log
```

Outputs:

```text
logs/train_dgf_dropout_smoke.log
results/dgf_dropout_smoke/train_log.csv
checkpoints/dgf_dropout_smoke/best_dgf_dropout.pt
```

Result:

```text
seed: 114514
best epoch: 20
dev loss: 1.6881
dev accuracy: 0.5582
dev weighted F1: 0.5691
dev macro F1: 0.4269
```

Current smoke comparison:

```text
text-only weighted F1: 0.5299
concat T+A+V weighted F1: 0.5464
DGF weighted F1: 0.5695
DGF + modality dropout weighted F1: 0.5691
```

In this short smoke setting, modality dropout is almost tied with plain DGF. A longer formal run may still be useful because modality dropout is intended more for robustness than immediate dev-set gain.

## 2026-04-30: DGF with BiGRU Dialogue Context

Implemented the dialogue-level context model.

Main changes:

- Added dialogue batching in `src/feature_dataset.py`.
- Added `DGFContextClassifier` in `src/models/context.py`.
- The model first applies DGF to each utterance, then runs a 1-layer BiGRU over utterances in the same dialogue.
- Padding masks are used so loss and metrics are computed only on real utterances.
- Updated `src/train.py` so `use_context: true` switches from utterance batches to dialogue batches.

Smoke training command:

```text
conda run -n workspace python -u -m src.train --config configs/dgf_context.yaml --device cpu --max-epochs 20 --patience 5 --batch-size 8 --output-dir results/dgf_context_smoke --checkpoint-dir checkpoints/dgf_context_smoke 2>&1 | tee logs/train_dgf_context_smoke.log
```

Outputs:

```text
logs/train_dgf_context_smoke.log
results/dgf_context_smoke/train_log.csv
checkpoints/dgf_context_smoke/best_dgf_context.pt
```

Result:

```text
seed: 114514
best epoch: 11
dev loss: 1.5387
dev accuracy: 0.5555
dev weighted F1: 0.5638
dev macro F1: 0.4343
```

Current smoke comparison:

```text
text-only weighted F1: 0.5299
concat T+A+V weighted F1: 0.5464
DGF weighted F1: 0.5695
DGF + modality dropout weighted F1: 0.5691
DGF + modality dropout + BiGRU context weighted F1: 0.5638
```

In this short smoke setting, the context model is slightly below plain DGF. The macro F1 is a bit higher than the previous DGF variants, so it is still worth keeping for the formal run and later analysis.

## 2026-04-30: Formal GPU Training on Dev Split

Ran the five Stage-A experiments with GPU using the full config settings:

- `max_epochs: 50`
- `early_stopping_patience: 5`
- `seed: 114514`
- model selection by dev weighted F1

Commands used the same pattern:

```text
conda run -n workspace python -u -m src.train --config configs/{config}.yaml --device cuda --output-dir results/{model} --checkpoint-dir checkpoints/{model} 2>&1 | tee logs/train_{model}.log
```

Outputs:

```text
logs/train_text_only.log
logs/train_concat_tav.log
logs/train_dgf.log
logs/train_dgf_dropout.log
logs/train_dgf_context.log

results/{model}/train_log.csv
checkpoints/{model}/best_{model}.pt
```

Formal GPU dev results:

```text
text_only      best_epoch=25  accuracy=0.5176  weighted_f1=0.5402  macro_f1=0.4050
concat_tav     best_epoch=11  accuracy=0.5176  weighted_f1=0.5351  macro_f1=0.3883
dgf            best_epoch=25  accuracy=0.5708  weighted_f1=0.5734  macro_f1=0.4295
dgf_dropout    best_epoch=22  accuracy=0.5582  weighted_f1=0.5675  macro_f1=0.4149
dgf_context    best_epoch=4   accuracy=0.5293  weighted_f1=0.5422  macro_f1=0.4021
```

Best formal dev result so far:

```text
DGF, dev weighted F1 = 0.5734
```

The context model underperformed in this run, likely because the BiGRU adds more parameters and overfits quickly under the current settings. It may need smaller hidden size, stronger dropout, or a lower learning rate before it becomes useful.

## 2026-04-30: Test Evaluation

Implemented `src/evaluate.py` and evaluated the five formal checkpoints on the official MELD test split.

Evaluation outputs are saved under:

```text
results/evaluate/{model}_test/
  metrics.json
  predictions.csv
  confusion_matrix.csv
  confusion_matrix.png
```

Test results:

```text
text_only      accuracy=0.5487  weighted_f1=0.5722  macro_f1=0.4158
concat_tav     accuracy=0.5563  weighted_f1=0.5794  macro_f1=0.4067
dgf            accuracy=0.5613  weighted_f1=0.5740  macro_f1=0.4020
dgf_dropout    accuracy=0.5770  weighted_f1=0.5903  macro_f1=0.4263
dgf_context    accuracy=0.5571  weighted_f1=0.5732  macro_f1=0.4081
```

Best test result so far:

```text
DGF + modality dropout, test weighted F1 = 0.5903
```

This is a more encouraging result than the dev-only comparison. The best dev model was plain DGF, but the best test model is DGF with modality dropout, which suggests the dropout variant may generalize a little better even when its dev score is slightly lower.

Evaluation commands followed this pattern:

```text
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/{config}.yaml --checkpoint checkpoints/{model}/best_{model}.pt --split test --device cuda --output-dir results/evaluate/{model}_test 2>&1 | tee logs/evaluate_{model}_test.log
```

The direct environment Python path was used for the final evaluation commands because `conda run -n workspace` intermittently hit import issues in this environment.

## 2026-04-30: Visualization and Gate Weight Export

Completed the first visualization and model-analysis export pass.

Main code changes:

- Updated `src/evaluate.py` so DGF-style models save utterance-level gate weights during evaluation.
- Updated `src/train.py` so future training runs save both `best_{model}.pt` and `last_{model}.pt`.
- Implemented `src/visualize.py` for report-ready plots from saved evaluation outputs.

Gate weight files were generated for:

```text
results/evaluate/dgf_test/gate_weights.csv
results/evaluate/dgf_dropout_test/gate_weights.csv
results/evaluate/dgf_context_test/gate_weights.csv
```

Visualization outputs:

```text
results/visualizations/f1_comparison.png
results/visualizations/confusion_matrix_dgf_dropout.png
results/visualizations/per_class_f1_dgf_dropout.png
results/visualizations/gate_weights_by_emotion_dgf_dropout.png
results/visualizations/metrics_summary.csv
results/visualizations/gate_weights_by_emotion_dgf_dropout.csv
```

The current best test model is still `dgf_dropout`, so the detailed confusion matrix, per-class F1 plot, and gate-weight plot use that model by default.

Commands:

```text
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/dgf_dropout.yaml --checkpoint checkpoints/dgf_dropout/best_dgf_dropout.pt --split test --device cpu --output-dir results/evaluate/dgf_dropout_test 2>&1 | tee logs/evaluate_dgf_dropout_test.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/dgf.yaml --checkpoint checkpoints/dgf/best_dgf.pt --split test --device cpu --output-dir results/evaluate/dgf_test 2>&1 | tee logs/evaluate_dgf_test.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/dgf_context.yaml --checkpoint checkpoints/dgf_context/best_dgf_context.pt --split test --device cpu --output-dir results/evaluate/dgf_context_test 2>&1 | tee logs/evaluate_dgf_context_test.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.visualize --best-model dgf_dropout 2>&1 | tee logs/visualize_test.log
```

CUDA evaluation was attempted first, but the current sandboxed command environment could not see the NVIDIA driver. Since this step only does forward-pass evaluation and plotting, CPU evaluation was used to generate the analysis files; the resulting metrics match the previous test scores.

Initial gate-weight observation:

```text
For dgf_dropout on test, the learned gate is heavily text-dominant.
Average text gate is roughly 0.79-0.92 across emotion classes.
```

This is useful evidence for the next performance-improvement discussion: the current multimodal model is not using audio/visual features strongly enough, so simply adding modalities is unlikely to reach a much higher score without improving feature quality, fusion pressure, or fine-tuning strategy.

Missing-modality evaluation was also added through `--zero-modality` in `src.evaluate`, then plotted by `src.visualize`.

Missing-modality results for `dgf_dropout` on test:

```text
full             accuracy=0.5770  weighted_f1=0.5903  macro_f1=0.4263
no_text          accuracy=0.2812  weighted_f1=0.2857  macro_f1=0.1353
no_audio         accuracy=0.5893  weighted_f1=0.5958  macro_f1=0.4333
no_visual        accuracy=0.5916  weighted_f1=0.5977  macro_f1=0.4322
no_audio_visual  accuracy=0.5927  weighted_f1=0.5975  macro_f1=0.4300
```

Additional output:

```text
results/visualizations/missing_modality_analysis_dgf_dropout.png
results/visualizations/missing_modality_metrics_dgf_dropout.csv
```

The ablation result is important: removing audio and/or visual features slightly improves the current model. This strongly suggests that the current audio/visual frozen features are adding noise rather than useful complementary signal.

## 2026-05-06: Strong Text Fine-tuning and Code Cleanup

After the initial multimodal DGF experiments, the main bottleneck was feature quality rather than the fusion layer itself. The missing-modality analysis showed that removing audio and/or visual features slightly improved `dgf_dropout`, so the first performance push focused on making the strongest text branch substantially better.

Main code changes:

- Added `src/train_text_finetune.py` for end-to-end Transformer text encoder fine-tuning.
- Added speaker-aware dialogue-context text construction with configurable `context_window`.
- Added `src/evaluate_text_ensemble.py` for logits/probability ensemble evaluation.
- Generalized the fine-tune classifier naming from RoBERTa-specific to Transformer-style, while preserving checkpoint compatibility.
- Added `sentencepiece` to `requirements.txt` for DeBERTa-v3 tokenization.
- Added `.codex` to `.gitignore` so local agent metadata does not show up as an untracked project file.

Fine-tuned text results on MELD test:

```text
frozen DGF dropout baseline          weighted_f1=0.5903  macro_f1=0.4263
RoBERTa-base single utterance        weighted_f1=0.6124  macro_f1=0.4569
RoBERTa-base context window 3        weighted_f1=0.6466  macro_f1=0.4677
RoBERTa-base context window 5        weighted_f1=0.6506  macro_f1=0.4815
RoBERTa-base context window 8        weighted_f1=0.6395  macro_f1=0.4446
RoBERTa-large context window 5       weighted_f1=0.6594  macro_f1=0.5118
DeBERTa-v3-base context window 5     weighted_f1=0.6557  macro_f1=0.4791
tuned text ensemble                  weighted_f1=0.6744  macro_f1=0.5184
```

Best result so far:

```text
Tuned text ensemble, test weighted F1 = 0.6744
```

This is a clear improvement over the frozen-feature multimodal baseline, but it also confirms that the current audio/visual branch is not strong enough yet. For the project narrative, this text result should be treated as a strong upper baseline or teacher model, not as the final multimodal answer.

Multimodal plan from here:

- Run audio-only and visual-only baselines before trusting any fusion improvement.
- Replace Wav2Vec2 mean-pooled frozen audio with stronger utterance-level speech features such as Whisper or HuBERT, or use a better temporal pooling scheme.
- Replace uniform-frame CLIP visual features with face-centered frames or an expression-focused visual encoder.
- Try late fusion over strong unimodal logits before returning to DGF, because the current feature-level fusion lets noisy modalities hurt the text branch.
- Add quality-aware gating: missing media flags, audio/visual availability, and possibly per-modality confidence should influence fusion weights.
- Consider multimodal distillation where the strong text ensemble acts as a teacher and audio/visual branches are trained to add complementary evidence rather than override text.
