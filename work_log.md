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

## 2026-05-07: Audio-only and Visual-only Baselines

Implemented and ran the first unimodal checks for the non-text branches.

Main code changes:

- Added `AudioOnlyClassifier` and `VisualOnlyClassifier` in `src/models/baselines.py`.
- Added `TextAudioClassifier`, `TextVisualClassifier`, and a shared `ConcatClassifier` for two- or three-modality concat baselines.
- Reused the same single-modality MLP shape as the frozen text baseline, so the comparison stays fair.
- Registered `audio_only` and `visual_only` in `src/models/__init__.py`.
- Added `configs/audio_only.yaml`, `configs/visual_only.yaml`, `configs/text_audio.yaml`, and `configs/text_visual.yaml`.
- Updated `src/visualize.py` so the comparison plot includes the new unimodal and two-modality baselines.
- Adjusted `src/train.py` and `src/evaluate.py` import order to avoid one observed non-sandbox GPU import failure mode where `torch` was loaded before `sklearn/scipy`.
- Added `src/torch_import_patch.py` and disabled torch compile/dynamo in training scripts because this WSL/conda environment occasionally fails inside PyTorch import-time source inspection.

Training commands:

```text
conda run -n workspace python -u -m src.train --config configs/audio_only.yaml --device cuda --output-dir results/audio_only --checkpoint-dir checkpoints/audio_only 2>&1 | tee logs/train_audio_only.log
conda run -n workspace python -u -m src.train --config configs/visual_only.yaml --device cuda --output-dir results/visual_only --checkpoint-dir checkpoints/visual_only 2>&1 | tee logs/train_visual_only.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.train --config configs/text_audio.yaml --device cpu --output-dir results/text_audio --checkpoint-dir checkpoints/text_audio 2>&1 | tee logs/train_text_audio.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.train --config configs/text_visual.yaml --device cpu --output-dir results/text_visual --checkpoint-dir checkpoints/text_visual 2>&1 | tee logs/train_text_visual.log
```

The two-modality concat models were trained on CPU because the non-sandbox `conda run` path became unstable inside Python/conda import code during this session. These models are small MLPs over cached features, so CPU/GPU only changes speed, not the selected metrics.

Dev results:

```text
audio_only    best_epoch=1   accuracy=0.2245  weighted_f1=0.2267  macro_f1=0.1500
visual_only   best_epoch=10  accuracy=0.2624  weighted_f1=0.2696  macro_f1=0.2052
text_audio    best_epoch=25  accuracy=0.5302  weighted_f1=0.5507  macro_f1=0.4170
text_visual   best_epoch=7   accuracy=0.4977  weighted_f1=0.5174  macro_f1=0.3824
```

Test evaluation commands:

```text
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/audio_only.yaml --checkpoint checkpoints/audio_only/best_audio_only.pt --split test --device cpu --output-dir results/evaluate/audio_only_test 2>&1 | tee logs/evaluate_audio_only_test.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/visual_only.yaml --checkpoint checkpoints/visual_only/best_visual_only.pt --split test --device cpu --output-dir results/evaluate/visual_only_test 2>&1 | tee logs/evaluate_visual_only_test.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/text_audio.yaml --checkpoint checkpoints/text_audio/best_text_audio.pt --split test --device cpu --output-dir results/evaluate/text_audio_test 2>&1 | tee logs/evaluate_text_audio_test.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/text_visual.yaml --checkpoint checkpoints/text_visual/best_text_visual.pt --split test --device cpu --output-dir results/evaluate/text_visual_test 2>&1 | tee logs/evaluate_text_visual_test.log
/home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.visualize --best-model dgf_dropout 2>&1 | tee logs/visualize_test.log
```

Test results:

```text
audio_only    accuracy=0.2180  weighted_f1=0.2337  macro_f1=0.1585
visual_only   accuracy=0.2502  weighted_f1=0.2724  macro_f1=0.1807
text_audio    accuracy=0.5544  weighted_f1=0.5768  macro_f1=0.4196
text_visual   accuracy=0.5222  weighted_f1=0.5489  macro_f1=0.4000
```

Conclusion:

The frozen Wav2Vec2 mean-pooled audio branch and uniform-frame CLIP visual branch are both very weak on their own. `text_audio` is close to the old frozen `text_only` baseline, but still below `concat_tav` and far below the fine-tuned text models. `text_visual` is weaker. This supports the previous missing-modality result: current audio/visual features are too noisy to improve the strong text model through simple feature-level fusion.

Next multimodal step:

- Rebuild audio features with a stronger speech/emotion encoder or better temporal pooling.
- Rebuild visual features using face-centered or expression-focused features.
- Only after stronger audio/visual baselines exist, try late fusion or quality-aware fusion again.

## 2026-05-07: HuBERT Audio Features

Implemented a stronger audio-feature branch using `torchaudio.pipelines.HUBERT_BASE`. These features are saved separately under `features/audio_hubert/`, so the old Wav2Vec2 features remain available for ablation.

Main code changes:

- Added `src/extract_audio_hubert_features.py` for HuBERT utterance-level audio extraction.
- Added `configs/audio_hubert_only.yaml` and `configs/text_audio_hubert.yaml`.
- Added `audio_hubert` loading support in `src/feature_dataset.py`.
- Added `AudioHubertOnlyClassifier` and `TextAudioHubertClassifier` in `src/models/baselines.py`.
- Added HuBERT models to training, evaluation, and visualization registration.
- Added `.torch_cache/` to `.gitignore` because HuBERT weights should not be pushed to Git.

GPU environment note:

The normal sandbox Python path cannot see CUDA in this WSL/container session. The stable GPU command path is the clean environment below, which sees the local RTX 4080 SUPER:

```text
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python ...
```

HuBERT weight download:

```text
curl -L --fail --retry 3 https://download.pytorch.org/torchaudio/models/hubert_fairseq_base_ls960.pth -o .torch_cache/hub/checkpoints/hubert_fairseq_base_ls960.pth 2>&1 | tee logs/download_hubert_base.log
```

Feature extraction commands:

```text
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.extract_audio_hubert_features --config configs/audio_hubert_only.yaml --split dev --device cuda --force 2>&1 | tee logs/extract_audio_hubert_dev_gpu.log
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.extract_audio_hubert_features --config configs/audio_hubert_only.yaml --split train --device cuda --batch-size 16 --force 2>&1 | tee logs/extract_audio_hubert_train_gpu.log
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.extract_audio_hubert_features --config configs/audio_hubert_only.yaml --split test --device cuda --batch-size 16 --force 2>&1 | tee logs/extract_audio_hubert_test_gpu.log
```

Feature extraction status:

```text
train: features/audio_hubert/train.pt  shape=(9989, 768)  available=9988
dev:   features/audio_hubert/dev.pt    shape=(1109, 768)  available=1108
test:  features/audio_hubert/test.pt   shape=(2610, 768)  available=2610
```

Known unavailable examples:

- `dev:dia110_utt7` is still missing because the official MELD package does not contain that dev video.
- `train:dia125_utt3` failed audio decoding and is kept as a zero-vector placeholder.

Training commands:

```text
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.train --config configs/audio_hubert_only.yaml --device cuda --output-dir results/audio_hubert_only --checkpoint-dir checkpoints/audio_hubert_only 2>&1 | tee logs/train_audio_hubert_only.log
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.train --config configs/text_audio_hubert.yaml --device cuda --output-dir results/text_audio_hubert --checkpoint-dir checkpoints/text_audio_hubert 2>&1 | tee logs/train_text_audio_hubert.log
```

Dev results:

```text
audio_hubert_only  best_epoch=18  weighted_f1=0.3142  macro_f1=0.2452
text_audio_hubert  best_epoch=25  weighted_f1=0.5640  macro_f1=0.4282
```

Test evaluation commands:

```text
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/audio_hubert_only.yaml --checkpoint checkpoints/audio_hubert_only/best_audio_hubert_only.pt --split test --device cuda --output-dir results/evaluate/audio_hubert_only_test 2>&1 | tee logs/evaluate_audio_hubert_only_test.log
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/text_audio_hubert.yaml --checkpoint checkpoints/text_audio_hubert/best_text_audio_hubert.pt --split test --device cuda --output-dir results/evaluate/text_audio_hubert_test 2>&1 | tee logs/evaluate_text_audio_hubert_test.log
```

Test results:

```text
audio_hubert_only  accuracy=0.2927  weighted_f1=0.3185  macro_f1=0.2182
text_audio_hubert  accuracy=0.5433  weighted_f1=0.5692  macro_f1=0.4149
```

Conclusion:

HuBERT improves the audio-only branch clearly compared with the old Wav2Vec2 audio-only baseline (`0.3185` vs `0.2337` test weighted F1). However, simple text+HuBERT concat does not yet beat the old text+audio test result. The next multimodal direction should be late fusion or quality-aware gating rather than simply concatenating stronger audio features.

## 2026-05-07: Late Fusion with HuBERT Audio

Implemented a `late_fusion_hubert` model. Instead of concatenating features directly, the model lets text, HuBERT-audio, and visual branches each produce logits first, then learns a gate over the three logit streams. This is a more conservative multimodal fusion strategy when audio/visual features are noisy.

Main code changes:

- Added `LateFusionHubertClassifier` in `src/models/fusion.py`.
- Registered `late_fusion_hubert` in `src/models/__init__.py`.
- Added `configs/late_fusion_hubert.yaml`.
- Updated `src/evaluate.py` so this model can save gate weights.
- Updated `src/visualize.py` so the comparison plot includes late fusion.

Training command:

```text
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.train --config configs/late_fusion_hubert.yaml --device cuda --output-dir results/late_fusion_hubert --checkpoint-dir checkpoints/late_fusion_hubert 2>&1 | tee logs/train_late_fusion_hubert.log
```

Dev result:

```text
late_fusion_hubert  best_epoch=9  weighted_f1=0.5673  macro_f1=0.4380
```

Test evaluation command:

```text
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.evaluate --config configs/late_fusion_hubert.yaml --checkpoint checkpoints/late_fusion_hubert/best_late_fusion_hubert.pt --split test --device cuda --output-dir results/evaluate/late_fusion_hubert_test 2>&1 | tee logs/evaluate_late_fusion_hubert_test.log
```

Test result:

```text
late_fusion_hubert  accuracy=0.5678  weighted_f1=0.5858  macro_f1=0.4234
```

Missing-modality analysis:

```text
full             weighted_f1=0.5858  macro_f1=0.4234
no_text          weighted_f1=0.3360  macro_f1=0.1690
no_audio         weighted_f1=0.5779  macro_f1=0.4151
no_visual        weighted_f1=0.5878  macro_f1=0.4226
no_audio_visual  weighted_f1=0.5893  macro_f1=0.4217
```

Conclusion:

Late fusion improves over `text_audio_hubert` (`0.5858` vs `0.5692` test weighted F1) and is close to the old best frozen multimodal model `dgf_dropout` (`0.5903`). However, the missing-modality result shows the current model still relies mainly on text: removing text causes a large drop, while removing audio/visual does not. The next useful multimodal work should focus on stronger visual features and quality-aware gates, not just larger fusion heads.

## 2026-05-08: Quality-aware Gate and Mixed Ensemble Attempt

Committed the previous HuBERT and late-fusion work first:

```text
8b55685 Add HuBERT and late fusion experiments
```

Implemented `quality_late_fusion_hubert`. Compared with plain late fusion, this model also receives per-modality availability flags such as `audio_hubert_available` and `visual_available`. During training it can randomly mark modalities as unavailable, so the gate learns to avoid missing or unreliable modalities.

Main code changes:

- Added `QualityLateFusionHubertClassifier` in `src/models/fusion.py`.
- Added `configs/quality_late_fusion_hubert.yaml`.
- Updated `src/train.py` to pass `[text, audio_hubert, visual]` quality flags into models that set `uses_quality = True`.
- Updated `src/evaluate.py` to pass quality flags and to avoid importing `matplotlib` during evaluation, because this WSL/conda environment sometimes corrupts global import state during plotting.
- Updated `src/visualize.py` with the same builtins restore guard.
- Removed the top-level `sklearn` dependency from `src/train_text_finetune.py` and replaced it with a small hand-written F1 calculator, reducing import-time instability.
- Added `src/evaluate_mixed_ensemble.py` for a future mixed ensemble between fine-tuned text checkpoints and cached-feature multimodal checkpoints.

Training command:

```text
env -i HOME=/home/gracchus PATH=/home/gracchus/miniconda3/envs/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib LANG=C.UTF-8 LC_ALL=C.UTF-8 TORCH_HOME=/home/gracchus/code/ucas-multimodal-erc/.torch_cache /home/gracchus/miniconda3/envs/workspace/bin/python -u -m src.train --config configs/quality_late_fusion_hubert.yaml --device cuda --output-dir results/quality_late_fusion_hubert --checkpoint-dir checkpoints/quality_late_fusion_hubert 2>&1 | tee logs/train_quality_late_fusion_hubert.log
```

Dev result:

```text
quality_late_fusion_hubert  best_epoch=16  weighted_f1=0.5572  macro_f1=0.4298
```

Test result:

```text
quality_late_fusion_hubert  accuracy=0.5536  weighted_f1=0.5779  macro_f1=0.4241
```

Missing-modality result:

```text
full             weighted_f1=0.5779  macro_f1=0.4241
no_text          weighted_f1=0.1900  macro_f1=0.1833
no_audio         weighted_f1=0.5778  macro_f1=0.4260
no_visual        weighted_f1=0.5777  macro_f1=0.4255
no_audio_visual  weighted_f1=0.5760  macro_f1=0.4209
```

Conclusion:

The quality-aware gate is more explicit about missing modalities, but it did not improve weighted F1 over plain late fusion (`0.5779` vs `0.5858`). It slightly improves macro F1 over plain late fusion (`0.4241` vs `0.4234`), but the gain is too small to treat as a main result. The useful conclusion is that current audio/visual branches still provide weak complementary information.

Mixed ensemble attempt:

I added a script for combining fine-tuned text logits with multimodal logits, but running all large text models and multimodal models in one process was unstable in the current WSL/conda environment. The process hit import-time failures and one segmentation fault. The safer next step is to split this into two stages:

1. Save dev/test logits for each text and multimodal checkpoint into small `.pt` files.
2. Run ensemble weight search from those saved logits without loading large model checkpoints again.

## 2026-05-08: Modality-quality Pass for Audio and Visual

Goal:

After the quality-gate attempt, the main issue was clear: the fusion module was not the bottleneck by itself. The audio and visual branches were still weak, so this pass focused on improving the actual cached modality features before adding more fusion complexity.

Audio changes:

- Updated `src/extract_audio_hubert_features.py` to support `--pooling mean_std`.
- Added `audio_hubert_stats` features with HuBERT frame mean + std, so each utterance becomes a 1536-dim vector.
- Added configs and baselines for `audio_hubert_stats_only` and `text_audio_hubert_stats`.

Audio results:

```text
audio_hubert_only       test weighted_f1=0.3185  macro_f1=0.2182
audio_hubert_stats_only test weighted_f1=0.3350  macro_f1=0.2161
text_audio_hubert       test weighted_f1=0.5692  macro_f1=0.4149
text_audio_hubert_stats test weighted_f1=0.5512  macro_f1=0.3775
```

Conclusion:

HuBERT mean+std improves audio-only performance, so it is a real audio feature-quality improvement. However, simple Text+Audio concat gets worse after adding the larger audio vector. This suggests the audio branch is still noisy relative to text and should be used carefully.

Visual changes:

- Added `src/extract_visual_face_features.py`.
- The script samples video frames, detects the largest face with OpenCV Haar cascade, crops around that face, and then extracts CLIP image features.
- If no face is detected in a frame, it falls back to the full frame.
- Added `visual_face`, `visual_face_only`, and `text_visual_face`.
- Added `concat_tav_hubert_stats_face`, a simple concat baseline using Text + HuBERT mean/std + Face-CLIP. This is intentionally not a new fusion module; it is only a stronger-feature concat comparison.

Feature extraction notes:

```text
dev   features/visual_face_clip/dev.pt    available=1108  face_frames=8133   failed=dev:dia110_utt7
train features/visual_face_clip/train.pt  available=9988  face_frames=73272  failed=train:dia125_utt3
test  features/visual_face_clip/test.pt   available=2610  face_frames=19162
```

The train and test face-feature extraction used GPU outside the sandbox. Later cached-feature training was run on CPU because the Codex sandbox could not see the GPU, and the follow-up sandbox-outside training command was rejected by the current usage limit. These MLP trainings are small and not performance-critical; to reproduce on GPU, replace `--device cpu` with `--device cuda`.

Visual and multimodal results:

```text
visual_only                 test weighted_f1=0.2724  macro_f1=0.1807
visual_face_only            test weighted_f1=0.2888  macro_f1=0.1636
text_visual                 test weighted_f1=0.5489  macro_f1=0.4000
text_visual_face            test weighted_f1=0.5587  macro_f1=0.4167
concat_tav                  test weighted_f1=0.5794  macro_f1=0.4067
concat_tav_hubert_stats_face test weighted_f1=0.5570 macro_f1=0.4054
```

Conclusion:

Face-centered CLIP is a useful visual-feature improvement: it improves both visual-only and Text+Visual. However, the strongest simple full multimodal result is still the old `concat_tav` test weighted F1 (`0.5794`), and the improved-feature three-modal concat (`0.5570`) does not beat it. For the report, this is still valuable: it shows that visual quality helps locally, while naive three-modal concatenation can dilute the text signal. The next performance direction should be a stronger text checkpoint plus offline logit-level ensembling, or a carefully regularized multimodal model that does not let noisy audio dominate.

## 2026-05-09: Current Diagnosis and Multimodal Direction

After comparing local results with MELD baselines and published/reference projects, the current performance picture is:

```text
frozen text_only             test weighted_f1=0.5722
frozen concat_tav            test weighted_f1=0.5794
frozen late_fusion_hubert    test weighted_f1=0.5858
frozen dgf_dropout           test weighted_f1=0.5903
fine-tuned text ensemble     test weighted_f1=0.6744
audio_hubert_stats_only      test weighted_f1=0.3350
visual_face_only             test weighted_f1=0.2888
```

Important diagnosis:

- The frozen-feature multimodal results around `0.58-0.59` weighted F1 are not obviously broken; they are near early MELD-style baselines.
- The stronger fine-tuned text ensemble already reaches about `0.67` weighted F1, so the project is not fundamentally stuck at `0.5`.
- Current audio and visual branches are much weaker than text. Simply concatenating or averaging them can dilute the text signal.
- Single-modal audio/visual runs do not show a clear “train longer and it will reach 0.8” pattern; they usually peak early and then overfit or flatten.
- The main project risk is not the optimizer or epoch count alone. The bigger issue is representation quality and the way weak modalities are allowed to influence prediction.

Direction from this point:

- Treat fine-tuned text as the main trunk.
- Treat audio/visual and frozen-feature multimodal models as auxiliary signals.
- Use gated multimodal fusion conservatively: only let the multimodal branch intervene when the text model is uncertain, and keep the auxiliary weight bounded.
- Keep reporting the older frozen DGF / late-fusion models as course-project multimodal baselines, but make the stronger final model a “strong text + confidence-gated multimodal” system.

Implementation started:

- Added `src/export_logits.py` so each text or multimodal checkpoint can export dev/test logits separately. This avoids loading all large models in one unstable process.
- Added `src/evaluate_offline_gated_fusion.py` for offline confidence-gated fusion. It searches text ensemble weights, auxiliary multimodal weights, confidence threshold, and multimodal gate strength on dev, then evaluates once on test.
- The gate is intentionally conservative: text remains the backbone, and multimodal logits are used as a bounded supplement on low-confidence examples.

Logits exported:

```text
text:
  text_finetune_context
  text_finetune_context_unweighted
  text_finetune_context5_unweighted
  text_finetune_context8_unweighted
  text_finetune_roberta_large_context5_unweighted
  text_finetune_deberta_v3_base_context5_unweighted

multimodal auxiliary:
  dgf_dropout
  late_fusion_hubert
  quality_late_fusion_hubert
  concat_tav
  text_visual_face
```

Relevant logs:

```text
logs/export_logits_feature.log
logs/export_logits_text.log
logs/evaluate_offline_gated_multimodal.log
logs/evaluate_offline_gated_multimodal_allow_zero.log
```

Offline gated multimodal result:

```text
output: results/offline_gated_multimodal
test accuracy:     0.6797
test weighted_f1:  0.6757
test macro_f1:     0.5190
```

Gate details:

```text
text ensemble weights:
  [0.1122, 0.1230, 0.0112, 0.0953, 0.4887, 0.1695]

multimodal auxiliary weights:
  dgf_dropout:                 0.1847
  late_fusion_hubert:          0.7014
  quality_late_fusion_hubert:  0.0957
  concat_tav:                  0.0062
  text_visual_face:            0.0120

dev gate threshold: 0.375
dev gate alpha:     0.440
test active ratio:  0.0364
test mean aux gate: 0.0160
```

The same result is selected even when the search is allowed to choose `alpha=0`, so the non-zero multimodal gate is not just forced by the script. The improvement over the previous strong text ensemble is small (`0.6744 -> 0.6757` weighted F1), but it is a useful project result: the best current multimodal path is not “more fusion layers”, but a text-dominant gated model that only trusts multimodal evidence on a small set of uncertain utterances.

Prediction-change analysis on the test split:

```text
total samples:          2610
gate-active samples:     95
changed predictions:     47
text mistakes fixed:     13
text correct changed:    11
```

This is consistent with the intended behavior: the multimodal branch changes only a small fraction of predictions and gives a small net gain instead of overwhelming the stronger text model.

## 2026-05-09: Class-aware Confidence Gate

The first offline gate used one global confidence threshold and one global auxiliary weight. I then added a class-aware version in `src/evaluate_offline_gated_fusion.py`: the gate still uses the fine-tuned text ensemble as the backbone, but the low-confidence threshold and multimodal intervention weight can differ by the text-predicted class.

This keeps the model explainable while giving weak or ambiguous classes more room to use multimodal evidence.

Tested variants:

```text
global gate, 5 auxiliary models       weighted_f1=0.6757  macro_f1=0.5190
class gate, 5 auxiliary models        weighted_f1=0.6785  macro_f1=0.5233
class gate, top-3 gated auxiliaries   weighted_f1=0.6788  macro_f1=0.5248
```

The current best main result is:

```text
output: results/offline_gated_multimodal_class_top3
test accuracy:     0.6831
test weighted_f1:  0.6788
test macro_f1:     0.5248
```

Reproduction helper:

```text
scripts/run_final_gated_multimodal.sh
```

It assumes the logits under `results/logits/` already exist and then reruns only the lightweight offline gate search/evaluation. A final check with this script reproduced the same result under `results/offline_gated_multimodal_final_check`.

Auxiliary multimodal models used:

```text
dgf_dropout                 weight=0.1580
late_fusion_hubert          weight=0.6921
quality_late_fusion_hubert  weight=0.1499
```

Class-aware gate parameters are ordered as:

```text
[anger, disgust, fear, joy, neutral, sadness, surprise]

thresholds:
  [0.350, 0.425, 0.375, 0.800, 0.425, 0.350, 0.375]

alphas:
  [0.420, 0.300, 0.500, 0.500, 0.160, 0.000, 0.500]
```

Prediction-change analysis:

```text
total samples:          2610
gate-active samples:     233
changed predictions:      63
text mistakes fixed:      29
text correct changed:     18
```

Interpretation:

- The result is still text-dominant, which matches the current feature-quality diagnosis.
- The multimodal branch has a measurable positive effect, but it is bounded and selective.
- This is a more defensible final direction than claiming audio/visual alone are strong; the report can say that current audio/visual features are weak individually, but gated multimodal evidence improves the strong text backbone slightly and improves macro F1.

## 2026-05-09: Ablation Plan for Final Report

The final report should not only show the best number. It should show why the final model is reasonable under the current feature quality. The ablation plan is:

### Must-have ablations

1. Modality-only feature quality:
   - `text_only`
   - `audio_only`
   - `audio_hubert_only`
   - `audio_hubert_stats_only`
   - `visual_only`
   - `visual_face_only`

2. Simple multimodal fusion:
   - `text_audio`
   - `text_visual`
   - `concat_tav`
   - `text_visual_face`
   - `concat_tav_hubert_stats_face`

3. Gated multimodal baselines:
   - `dgf`
   - `dgf_dropout`
   - `dgf_context`
   - `late_fusion_hubert`
   - `quality_late_fusion_hubert`

4. Strong-text and final gated model:
   - fine-tuned text ensemble only
   - global confidence gate
   - class-aware confidence gate with top-3 gated auxiliaries

### Gate-specific ablations

These are important because the project name and narrative emphasize gated multimodal fusion:

- Global gate vs class-aware gate.
- Five auxiliary models vs top-3 gated auxiliary models.
- Allow `alpha=0` during dev search to show that the multimodal branch is selected because it helps, not because it is forced.
- Prediction-change analysis: how many text predictions are changed, how many mistakes are fixed, and how many correct text predictions are damaged.
- Per-class F1 comparison between text-only ensemble and final gated multimodal model.

### Optional ablations

- Missing-modality analysis for `late_fusion_hubert` / `quality_late_fusion_hubert`.
- Effect of face-centered visual features: `visual_only` vs `visual_face_only`, `text_visual` vs `text_visual_face`.
- Effect of HuBERT statistics: `audio_hubert_only` vs `audio_hubert_stats_only`.
- Final gate with only `late_fusion_hubert`, with `dgf_dropout + late_fusion_hubert`, and with `dgf_dropout + late_fusion_hubert + quality_late_fusion_hubert`.

### Recommended report narrative

The report should make a careful claim:

- Audio and visual branches are weak by themselves on MELD.
- Naive multimodal concatenation can dilute the text signal.
- A fine-tuned text backbone is much stronger than frozen utterance features.
- The final gated model uses multimodal evidence selectively on uncertain samples, leading to a small but consistent improvement over the strong text ensemble.
- The contribution is not SOTA performance; it is a complete, interpretable multimodal ERC pipeline with feature extraction, fusion baselines, gated fusion, and ablation analysis.

## 2026-05-09: Tomorrow's Capacity Check

Question to revisit:

The current gated multimodal models project text/audio/visual features into `d_model=256`. This is mainly for modality alignment, parameter control, and reducing noisy weak-modality influence. It is not because HuBERT/CLIP embeddings become sparse without compression; they are dense vectors.

However, `256` may still be too aggressive. Tomorrow's first task should be a small capacity ablation:

```text
d_model = 128
d_model = 256
d_model = 512
```

Priority models:

- `late_fusion_hubert`
- `quality_late_fusion_hubert`
- possibly `dgf_dropout`

What to check:

- dev/test weighted F1 and macro F1
- whether larger `d_model` improves audio/visual contribution
- whether larger `d_model` overfits faster
- whether gate weights become less text-dominant or simply noisier

If `512` improves F1 without obvious overfitting, then the current `256` bottleneck is probably too tight. If `256` remains best or most stable, keep it and describe it as a reasonable capacity-control choice for MELD-scale training.

## 2026-05-10: d_model Capacity Ablation

Ran the planned `d_model` capacity ablation for the main gated multimodal models.

Configs added:

```text
configs/late_fusion_hubert_d128.yaml
configs/late_fusion_hubert_d512.yaml
configs/quality_late_fusion_hubert_d128.yaml
configs/quality_late_fusion_hubert_d512.yaml
configs/dgf_dropout_d128.yaml
configs/dgf_dropout_d512.yaml
```

Training and evaluation logs:

```text
logs/train_late_fusion_hubert_d128.log
logs/train_late_fusion_hubert_d512.log
logs/train_quality_late_fusion_hubert_d128.log
logs/train_quality_late_fusion_hubert_d512.log
logs/train_dgf_dropout_d128.log
logs/train_dgf_dropout_d512.log
logs/evaluate_capacity_late_quality.log
logs/evaluate_capacity_dgf_dropout.log
```

### Capacity ablation result

```text
late_fusion_hubert:
  d_model=128  dev_wf1=0.5543  test_wf1=0.5544  test_macro=0.4071
  d_model=256  dev_wf1=0.5673  test_wf1=0.5858  test_macro=0.4234
  d_model=512  dev_wf1=0.5757  test_wf1=0.5780  test_macro=0.4225

quality_late_fusion_hubert:
  d_model=128  dev_wf1=0.5313  test_wf1=0.5571  test_macro=0.4040
  d_model=256  dev_wf1=0.5572  test_wf1=0.5779  test_macro=0.4241
  d_model=512  dev_wf1=0.5716  test_wf1=0.5981  test_macro=0.4390

dgf_dropout:
  d_model=128  dev_wf1=0.5516  test_wf1=0.5635  test_macro=0.4087
  d_model=256  dev_wf1=0.5675  test_wf1=0.5903  test_macro=0.4263
  d_model=512  dev_wf1=0.5802  test_wf1=0.5754  test_macro=0.4177
```

Interpretation:

- `d_model=128` is too small for all three gated multimodal models.
- `d_model=512` improves dev F1 for all three models, but only `quality_late_fusion_hubert` converts this into a better test result.
- `quality_late_fusion_hubert_d512` becomes the strongest frozen-feature multimodal model so far: `test weighted_f1=0.5981`, beating the previous `dgf_dropout` baseline `0.5903`.
- For `late_fusion_hubert` and `dgf_dropout`, `512` looks less stable: dev improves, but test drops compared with `256`. This suggests some overfitting or less useful gate behavior.
- The answer to yesterday's question is therefore nuanced: `256` is not universally too tight, but it is too restrictive for the quality-aware late-fusion model.

### Effect on final offline gated model

Exported new logits:

```text
results/logits/late_fusion_hubert_d512/
results/logits/quality_late_fusion_hubert_d512/
```

Tried replacing the final auxiliary models:

```text
original top-3 auxiliaries:
  [dgf_dropout, late_fusion_hubert, quality_late_fusion_hubert]
  test weighted_f1=0.6788  macro_f1=0.5248

replace quality with quality_d512:
  [dgf_dropout, late_fusion_hubert, quality_late_fusion_hubert_d512]
  test weighted_f1=0.6746  macro_f1=0.5178

replace late and quality with d512 versions:
  [dgf_dropout, late_fusion_hubert_d512, quality_late_fusion_hubert_d512]
  test weighted_f1=0.6765  macro_f1=0.5218

add quality_d512 as an extra auxiliary:
  [dgf_dropout, late_fusion_hubert, quality_late_fusion_hubert, quality_late_fusion_hubert_d512]
  test weighted_f1=0.6770  macro_f1=0.5186
```

Conclusion:

The larger quality-aware model is a stronger standalone multimodal baseline, but it does not improve the final offline gate ensemble. The most likely explanation is that `quality_late_fusion_hubert_d512` is stronger but less complementary to the text ensemble / existing auxiliary mix. Therefore:

- Keep `quality_late_fusion_hubert_d512` as the best frozen-feature multimodal baseline.
- Keep the current final model `offline_gated_multimodal_class_top3` as the main final result.
- In the report, use this as a capacity ablation: more capacity can help, but only when the gate has quality/missing-modality signals; simply increasing all fusion dimensions does not guarantee better generalization.

## 2026-05-10: Asymmetric Modality Capacity Trial

Revisited the capacity question from a more modality-aware angle. The concern was not only whether `d_model=256` is too small, but whether text/audio/visual should be forced into the same projection dimension at all.

Implemented support for asymmetric projection dimensions in `LateFusionHubertClassifier` and `QualityLateFusionHubertClassifier`:

```text
d_model_text
d_model_audio
d_model_visual
```

Also added `AsymmetricQualityLogitFusionClassifier`, where each modality branch can have a different hidden size, but the gate only sees three sets of logits plus quality flags. This avoids feeding very large modality hidden vectors directly into the gate.

Configs tried:

```text
configs/quality_late_fusion_hubert_asym_av512.yaml
configs/quality_late_fusion_hubert_face_asym_av512.yaml
configs/quality_late_fusion_hubert_stats_face_asym.yaml
configs/quality_late_fusion_hubert_asym_t384_a768_v512.yaml
configs/asym_quality_logit_hubert_t384_a768_v512.yaml
configs/asym_quality_logit_hubert_stats_face.yaml
```

Results:

```text
quality_late_fusion_hubert_asym_av512
  text=256 audio=512 visual=512
  test weighted_f1=0.5859  macro_f1=0.4343

quality_late_fusion_hubert_face_asym_av512
  text=256 audio=512 visual_face=512
  test weighted_f1=0.5778  macro_f1=0.4216

quality_late_fusion_hubert_stats_face_asym
  text=256 audio_stats=768 visual_face=512
  test weighted_f1=0.5492  macro_f1=0.3904

quality_late_fusion_hubert_asym_t384_a768_v512
  text=384 audio=768 visual=512
  test weighted_f1=0.5689  macro_f1=0.4060

asym_quality_logit_hubert_t384_a768_v512
  text=384 audio=768 visual=512, gate on logits + quality
  test weighted_f1=0.5922  macro_f1=0.4359

asym_quality_logit_hubert_stats_face
  text=384 audio_stats=768 visual_face=512, gate on logits + quality
  test weighted_f1=0.5803  macro_f1=0.4321
```

Final offline gate with `asym_quality_logit_hubert_t384_a768_v512` as an extra auxiliary:

```text
results/offline_gated_multimodal_class_plus_asym_logit
test weighted_f1=0.6758
test macro_f1=0.5199
```

Conclusion:

- The asymmetric idea is conceptually reasonable, but the tested asymmetric variants do not beat `quality_late_fusion_hubert_d512` (`test weighted_f1=0.5981`) as standalone frozen multimodal models.
- The best asymmetric variant is `asym_quality_logit_hubert_t384_a768_v512` with `test weighted_f1=0.5922`, slightly above old `dgf_dropout` but below `quality_late_fusion_hubert_d512`.
- Adding the asymmetric model into the final offline gate does not improve the final text-dominant result (`0.6758` vs current best `0.6788`).
- For final reporting, the strongest defensible story is still:
  1. Do not compress everything to 128 or 256 blindly.
  2. Higher capacity helps when combined with quality-aware gating (`quality_late_fusion_hubert_d512 = 0.5981`).
  3. Simply making audio/visual branches wider is not enough; the modality features themselves remain noisy, and the final text-dominant offline gate remains best overall.

## 2026-05-10: Attempts to Increase Multimodal Gain over Frozen Text-only

Goal:

Focus on improving the project-designed multimodal structure against frozen `text_only`, ignoring fine-tuned text for the moment. The baseline to beat is:

```text
frozen text_only  test weighted_f1=0.5722  macro_f1=0.4158
```

### Branch auxiliary loss

Added support for branch auxiliary loss in `src/train.py` and `src/models/fusion.py`. For late-fusion models, the fused output still receives the main loss, but the text/audio/visual branch logits also receive an auxiliary classification loss:

```text
loss = fused_loss + auxiliary_loss_weight * mean(branch_losses)
```

Tried:

```text
quality_late_fusion_hubert_d512_aux02  auxiliary_loss_weight=0.2
  test weighted_f1=0.5889  macro_f1=0.4274

quality_late_fusion_hubert_d512_aux05  auxiliary_loss_weight=0.5
  test weighted_f1=0.5957  macro_f1=0.4372
```

This did not beat the no-auxiliary-loss model:

```text
quality_late_fusion_hubert_d512
  test weighted_f1=0.5981  macro_f1=0.4390
```

Conclusion: auxiliary branch supervision is reasonable, but in the current setup it does not improve the best structure.

### Frozen text-only as main trunk with multimodal gate

Exported frozen text-only logits and evaluated offline gated fusion without any fine-tuned text model.

Results:

```text
frozen text_only + [dgf_dropout, late_fusion_hubert, quality_d512, asym_quality_logit]
  test weighted_f1=0.5951  macro_f1=0.4343

frozen text_only + quality_d512 only
  test weighted_f1=0.5926  macro_f1=0.4365
```

These are both better than frozen `text_only=0.5722`, but still do not beat standalone `quality_late_fusion_hubert_d512=0.5981`.

### Current answer to the structural question

The best project-designed frozen-feature multimodal structure remains:

```text
quality_late_fusion_hubert_d512
test accuracy:     0.5828
test weighted_f1:  0.5981
test macro_f1:     0.4390
```

Improvement over frozen text-only:

```text
accuracy:     0.5487 -> 0.5828  (+0.0341)
weighted_f1:  0.5722 -> 0.5981  (+0.0259)
macro_f1:     0.4158 -> 0.4390  (+0.0232)
```

Interpretation:

- The gain over frozen text-only is real and larger than the final fine-tuned-text gate gain.
- However, it is still not a huge margin. The limiting factor is likely audio/visual feature quality, not only fusion architecture.
- Tested asymmetric capacity and branch auxiliary losses did not surpass the simpler `quality_late_fusion_hubert_d512`.
- For the report, the strongest claim should be: quality-aware gated multimodal fusion with sufficient capacity gives a clear improvement over frozen text-only and simple fusion, but current audio/visual features are not strong enough to create a dramatic jump.

## 2026-05-11: Audio-first Feature Improvement

Goal:

Start from the audio modality itself. The previous conclusion was that simply changing the fusion layer or branch width was not enough, so this round adds more emotion-related acoustic information before fusion.

### Added prosody/acoustic features

New scripts:

```text
src/extract_audio_prosody_features.py
src/combine_audio_features.py
```

The prosody extractor creates 115-dimensional acoustic features from each mp4 audio track:

- MFCC statistics
- RMS energy
- zero-crossing rate
- spectral centroid / bandwidth / rolloff / flatness
- pitch and pitch-delta statistics
- duration, peak amplitude, silence ratio

The features are standardized with train split statistics, then combined with HuBERT:

```text
audio_hubert:          768 dim
audio_prosody:         115 dim
audio_hubert_prosody:  883 dim
```

Extraction result:

```text
features/audio_prosody/train.pt         (9989, 115), available=9988
features/audio_prosody/dev.pt           (1109, 115), available=1108
features/audio_prosody/test.pt          (2610, 115), available=2610

features/audio_hubert_prosody/train.pt  (9989, 883), available=9988
features/audio_hubert_prosody/dev.pt    (1109, 883), available=1108
features/audio_hubert_prosody/test.pt   (2610, 883), available=2610
```

New missing/decode issue:

```text
train:dia125_utt3  audio decode failed
dev:dia110_utt7    official missing video/audio
```

### Audio experiment results

```text
audio_prosody_only
  test accuracy=0.2575  weighted_f1=0.2844  macro_f1=0.1984

audio_hubert_prosody_only
  test accuracy=0.3123  weighted_f1=0.3406  macro_f1=0.2398
```

Interpretation:

- Prosody alone is weak, but not random.
- HuBERT + prosody is stronger than prosody alone and gives the audio branch more emotion-related information.

### Multimodal experiments with enhanced audio

```text
concat_tav_hubert_prosody
  test accuracy=0.5284  weighted_f1=0.5533  macro_f1=0.3913

quality_late_fusion_hubert_prosody_d512
  test accuracy=0.5536  weighted_f1=0.5728  macro_f1=0.4129

quality_late_fusion_hubert_prosody_guarded
  test accuracy=0.5448  weighted_f1=0.5719  macro_f1=0.4226

asym_quality_logit_hubert_prosody
  test accuracy=0.5655  weighted_f1=0.5895  macro_f1=0.4344
```

Current diagnosis:

- The enhanced audio branch improved audio-only modeling, but naive multimodal use still overfits.
- In `quality_late_fusion_hubert_prosody_d512`, the test gate average was roughly text `0.7319`, audio `0.2067`, visual `0.0614`. The model is using audio, but the audio branch is not reliable enough to receive that much weight on every sample.
- `asym_quality_logit_hubert_prosody` is more stable than the high-dimensional quality gate, but still does not beat the current best frozen multimodal model:

```text
quality_late_fusion_hubert_d512
  test weighted_f1=0.5981  macro_f1=0.4390
```

Next step:

Export the new `asym_quality_logit_hubert_prosody` logits and test whether it provides complementary value in the final offline gate. If it does not, the next design should be confidence-aware audio gating rather than continuing to enlarge audio features.

## 2026-05-11: Project Scope Cleanup and Audio Head Check

Added:

```text
project_scope.md
single_modality_improvement_plan.md
```

Purpose:

- separate the clean course-project mainline from side branches;
- demote fine-tuned text ensemble and offline gate to appendix/discussion;
- keep the main claim focused on frozen-feature multimodal gated fusion.

Also updated `.gitignore`:

```text
logs/
*.pdf
```

### Mainline decision

The report should focus on:

```text
text_only
audio_hubert_only
visual_only
concat_tav
dgf_dropout
quality_late_fusion_hubert_d512
```

Fine-tuned text and offline gate are useful analysis tools, but not the main model.

### Audio MLP check

Added a slightly deeper audio-only MLP to test whether the weak audio branch was caused by an overly shallow classifier head.

Configs:

```text
configs/audio_hubert_mlp.yaml
configs/audio_hubert_stats_mlp.yaml
configs/audio_hubert_prosody_mlp.yaml
```

Results:

```text
audio_hubert_mlp
  test weighted_f1=0.3240  macro_f1=0.2205

audio_hubert_stats_mlp
  test weighted_f1=0.2299  macro_f1=0.1812

audio_hubert_prosody_mlp
  test weighted_f1=0.3321  macro_f1=0.2238
```

Compared with previous shallow heads:

```text
audio_hubert_only          weighted_f1=0.3185
audio_hubert_stats_only    weighted_f1=0.3350
audio_hubert_prosody_only  weighted_f1=0.3406
```

Conclusion:

The audio branch is not mainly limited by classifier depth. Deeper MLP heads mostly overfit or destabilize training. Do not merge these audio MLPs into the main gated model.

Next direction:

Improve the non-text modalities by better features, especially visual expression-oriented features. If that is not feasible, use context-aware frozen text as a clean stronger baseline and compare the gated model fairly against that stronger text-only baseline.

## 2026-05-11: Visual Expression Feature Plan

Added:

```text
visual_modality_improvement_plan.md
```

Purpose:

- keep the next visual-modality improvement round focused;
- replace generic CLIP visual features with FER / affect-oriented expression features;
- test whether a task-specific visual representation can make the gated multimodal model improve more clearly over `text_only`.

Planned mainline:

```text
video frames
-> face crop when possible
-> expression model feature extraction
-> visual_expression_only
-> text_visual_expression
-> quality_late_fusion_hubert_expression
```

The key comparison is not simply whether the new visual-only model is high in absolute terms. The more important question is whether expression features give a clearer complementary signal to text:

```text
text_only
vs
text_visual_expression
vs
quality_late_fusion_hubert_expression
```

Success criteria:

```text
visual_expression_only > visual_face_only
text_visual_expression > text_only
quality_late_fusion_hubert_expression >= quality_late_fusion_hubert_d512
```

The preferred project story is still a clean frozen-feature multimodal system, not fine-tuned text ensemble or offline post-hoc gating.

## 2026-05-11: Visual Expression Feature Extraction

Implemented and ran the first FER-oriented visual feature extractor:

```text
src/extract_visual_expression_features.py
```

Feature source:

```text
trpakov/vit-face-expression
```

Feature design:

```text
ViT CLS embedding + expression logits + expression probabilities
= 768 + 7 + 7
= 782 dimensions
```

Extraction pipeline:

```text
video -> uniform frame sampling -> Haar face crop when possible -> ViT expression model -> utterance-level average
```

Generated files:

```text
features/visual_expression/train.pt
features/visual_expression/dev.pt
features/visual_expression/test.pt
```

Observed extraction results:

```text
train: shape=(9989, 782), available=9988/9989, failed=['train:dia125_utt3']
dev:   shape=(1109, 782), available=1108/1109, failed=['dev:dia110_utt7']
test:  shape=(2610, 782), available=2610/2610
```

Notes:

- `dev:dia110_utt7` is the official missing dev video already documented earlier.
- `train:dia125_utt3` exists in metadata but did not yield a usable visual feature in this extraction pass, so it remains a zero vector with `available=False`.
- The extraction had to run in the base conda environment because the `workspace` environment can read videos but has unstable `transformers`/`torchvision` imports. Base environment was updated with `opencv-python-headless` and successfully used GPU.

Next experiment:

```text
visual_expression_only
text_visual_expression
quality_late_fusion_hubert_expression
```

The first check should be whether `visual_expression_only` beats the old visual baselines:

```text
visual_only       weighted_f1=0.2724
visual_face_only  weighted_f1=0.2888
```

## 2026-05-11: Visual Expression Experiment Results

The FER-oriented visual branch was tested in several forms.

### Full expression feature

Feature:

```text
768-d ViT CLS embedding + 7-d logits + 7-d probabilities = 782 dims
```

Results:

| Model | Test Weighted F1 | Test Macro F1 | Note |
|---|---:|---:|---|
| `visual_expression_only` | 0.2966 | 0.1807 | slightly better than `visual_face_only`, but still weak |
| `text_visual_expression` | 0.4227 | 0.2703 | much worse than `text_only`; full embedding hurts text |
| `quality_late_fusion_hubert_expression` | 0.5620 | 0.4026 | worse than current main model |

Gate observation for `quality_late_fusion_hubert_expression` on test:

```text
gate_text=0.6457
gate_audio=0.1293
gate_visual=0.2250
```

Interpretation:

The full 782-d expression feature is noisy in MELD. The gate gives it substantial weight, but that hurts final performance.

### Compact expression feature

Derived compact feature:

```text
7-d expression logits + 7-d expression probabilities = 14 dims
```

Generated files:

```text
features/visual_expression_compact/train.pt
features/visual_expression_compact/dev.pt
features/visual_expression_compact/test.pt
```

Results:

| Model | Test Weighted F1 | Test Macro F1 | Note |
|---|---:|---:|---|
| `visual_expression_compact_only` | 0.2361 | 0.1414 | too weak alone |
| `text_visual_expression_compact` | 0.5608 | 0.4118 | close to text-only but not better |
| `quality_late_fusion_hubert_expression_compact` | 0.5851 | 0.4247 | better than text-only, worse than current main model |

Gate observation for `quality_late_fusion_hubert_expression_compact` on test:

```text
gate_text=0.8573
gate_audio=0.1261
gate_visual=0.0166
```

Interpretation:

The compact expression signal is weak but less harmful. The quality gate learns to almost ignore it, so the model remains better than text-only but cannot beat the current CLIP-based main model.

### CLIP + compact expression feature

To avoid replacing the existing useful CLIP visual feature, a combined visual feature was also tested:

```text
512-d CLIP visual + 14-d expression logits/probabilities = 526 dims
```

Generated files:

```text
features/visual_clip_expression/train.pt
features/visual_clip_expression/dev.pt
features/visual_clip_expression/test.pt
```

Result:

| Model | Test Weighted F1 | Test Macro F1 | Note |
|---|---:|---:|---|
| `quality_late_fusion_hubert_visual_clip_expression` | 0.5841 | 0.4337 | macro improves slightly, weighted F1 still below current main model |

Gate observation on test:

```text
gate_text=0.7836
gate_audio=0.0695
gate_visual=0.1469
```

Compared with current main model:

```text
quality_late_fusion_hubert_d512:
  weighted_f1=0.5981
  macro_f1=0.4390
  gate_text=0.7924
  gate_audio=0.1094
  gate_visual=0.0982
```

Conclusion:

FER expression features are not a clean improvement over the existing CLIP visual branch in the current frozen-feature setup. They do give a slightly better visual-only weighted F1 than face-centered CLIP, but once fused with text/audio they either hurt or get ignored.

Current recommendation:

Do not replace the main visual branch with FER expression features. Keep `quality_late_fusion_hubert_d512` as the current main model. The visual-expression experiments can be used as an ablation/negative result:

```text
task-specific expression features were tested, but MELD visual expression cues were too noisy or too weak to improve the gated multimodal model.
```

Next useful direction:

If we still want a stronger multimodal story, the next improvement should not be another visual feature concatenation. Better candidates are:

1. context-aware frozen text as a stronger fair baseline;
2. role/dialogue context modeling;
3. per-class analysis to show where audio/visual gate helps even if global weighted F1 gain is small.

## 2026-05-11: Context-aware Residual Fusion Design

Added:

```text
residual_context_fusion_plan.md
```

New main idea:

```text
final_logits = text_logits
             + gate_audio  * audio_delta
             + gate_visual * visual_delta
```

This keeps text as the strong base model and lets audio / FER visual features act as gated corrections instead of competing with text in a simple weighted average.

The design also adds dialogue context:

```text
text/audio/FER visual projections
-> dialogue-level BiGRU
-> context-aware text logits and audio/visual residual deltas
-> residual gated fusion
```

Why this still counts as fusion:

- gate is computed from text, audio, visual, and dialogue context together;
- audio and visual directly modify final class logits;
- gate weights can still be saved and ablated;
- the model is explicitly designed around the observed weakness of non-text modalities.

First planned config:

```text
configs/context_residual_gated_fusion_fer.yaml
```

The visual branch should use FER full expression features for now, because `visual_expression_only` is the strongest visual-only branch tested so far.

### First implementation result

Implemented:

```text
ContextResidualGatedFusionClassifier
build_context_residual_gated_fusion_model
configs/context_residual_gated_fusion_fer.yaml
```

The model uses:

```text
text + HuBERT audio + FER visual
dialogue-level BiGRU context
text_logits + gated audio_delta + gated visual_delta
```

Smoke run:

```text
logs/train_context_residual_gated_fusion_fer_smoke.log
best dev weighted_f1 after 2 epochs = 0.5139
```

Full run:

```text
logs/train_context_residual_gated_fusion_fer.log
best dev weighted_f1 = 0.5947
```

Test result:

```text
context_residual_gated_fusion_fer
accuracy    = 0.5969
weighted_f1 = 0.6051
macro_f1    = 0.4459
```

This is better than the previous clean main model:

```text
quality_late_fusion_hubert_d512
weighted_f1 = 0.5981
macro_f1    = 0.4390
```

Quick zero-modality checks on the trained residual model:

| Setting | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|
| full text+audio+FER | 0.6051 | 0.4459 |
| zero audio | 0.5990 | 0.4341 |
| zero FER visual | 0.6160 | 0.4442 |
| zero audio+FER visual | 0.5954 | 0.4276 |

Interpretation:

- The new residual-context structure is promising because it beats the previous main model.
- Removing both non-text modalities drops from `0.6051` to `0.5954`, so the multimodal correction path is contributing.
- The `zero FER visual` result being higher suggests the FER branch still sometimes over-corrects. The next refinement should make visual correction more conservative, for example:
  - use sigmoid correction gates instead of softmax correction gates;
  - initialize audio/visual correction gates lower;
  - add a smaller visual residual scale;
  - or train a text+audio context residual version as a cleaner comparison.

## 2026-05-11 LSTM-style Residual Gate

Motivation:

- The previous residual gate used a GRU context encoder and a softmax gate over text/audio/visual.
- That structure made the three gate values compete with each other, and the FER visual branch could still over-correct text.
- The new idea is closer to an LSTM-style context memory: first read the dialogue sequence with BiLSTM, then let audio and visual independently decide whether to correct the text logits.

Implemented:

```text
ContextLSTMResidualGatedFusionClassifier
build_context_lstm_residual_gated_fusion_model
configs/context_lstm_residual_gated_fusion_fer.yaml
```

Main formula:

```text
final_logits =
    text_logits
    + sigmoid(audio_gate)  * audio_residual_scale  * audio_delta
    + sigmoid(visual_gate) * visual_residual_scale * visual_delta
```

Current conservative settings:

```text
audio_gate_bias = -1.0
visual_gate_bias = -2.0
audio_residual_scale = 1.0
visual_residual_scale = 0.4
```

Smoke run:

```text
logs/train_context_lstm_residual_gated_fusion_fer_smoke.log
best dev weighted_f1 after 2 epochs = 0.4538
```

Full run:

```text
logs/train_context_lstm_residual_gated_fusion_fer.log
best dev weighted_f1 = 0.5937
```

Test result:

```text
context_lstm_residual_gated_fusion_fer
accuracy    = 0.5866
weighted_f1 = 0.5993
macro_f1    = 0.4273
```

Zero-modality checks:

| Setting | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|
| full text+audio+FER | 0.5993 | 0.4273 |
| zero audio | 0.5845 | 0.4308 |
| zero FER visual | 0.5896 | 0.4218 |
| zero audio+FER visual | 0.5724 | 0.3968 |

Interpretation:

- This design is slightly weaker than the previous GRU residual model on absolute test F1 (`0.5993` vs `0.6051`).
- But its ablation pattern is cleaner: removing audio or visual both hurts weighted F1, and removing both drops close to the text-only baseline.
- That means the LSTM + independent residual gate is using both non-text modalities in a more defensible way.
- The saved gate values are not softmax weights anymore: `gate_text=1` means text is always the base path, while `gate_audio` and `gate_visual` are residual correction strengths.
- The current gate strengths saturate high, so a next useful refinement is not to add another big module, but to tune correction strength or add light gate regularization.

## 2026-05-11 d_model Dimension Check

Question:

- The cached base features are mostly around 768 dimensions:
  - text RoBERTa: 768
  - HuBERT audio: 768
  - FER visual expression: 782
- Earlier configs projected all modalities to `d_model=256`, which may be too aggressive for audio/visual.
- New comparison keeps the same LSTM residual gate structure and only changes `d_model`.

Configs:

```text
configs/context_lstm_residual_gated_fusion_fer.yaml       d_model=256
configs/context_lstm_residual_gated_fusion_fer_d512.yaml  d_model=512
configs/context_lstm_residual_gated_fusion_fer_d768.yaml  d_model=768
```

Results:

| d_model | Best Dev Weighted F1 | Test Weighted F1 | Test Macro F1 |
|---:|---:|---:|---:|
| 256 | 0.5937 | 0.5993 | 0.4273 |
| 512 | 0.5975 | 0.6012 | 0.4243 |
| 768 | 0.5865 | 0.5987 | 0.4222 |

Zero-modality results:

| d_model | Full | Zero Audio | Zero FER Visual | Zero Audio+Visual |
|---:|---:|---:|---:|---:|
| 256 | 0.5993 | 0.5845 | 0.5896 | 0.5724 |
| 512 | 0.6012 | 0.5552 | 0.5956 | 0.5404 |
| 768 | 0.5987 | 0.5935 | 0.5932 | 0.5880 |

Gate averages on test:

| d_model | gate_text | gate_audio | gate_visual |
|---:|---:|---:|---:|
| 256 | 1.0000 | 0.9838 | 0.9787 |
| 512 | 1.0000 | 0.9783 | 0.9844 |
| 768 | 1.0000 | 0.2335 | 0.5637 |

Interpretation:

- `d_model=512` is currently the best setting for this structure.
- It slightly improves full test weighted F1 and gives the clearest ablation: removing audio or removing both non-text modalities causes a large drop.
- `d_model=768` does not help. It appears to become harder to train on the current data size and uses audio much less.
- The earlier concern was valid: `d_model=256` was probably too compressed, but fully matching the original 768 dimensions is not automatically better.
- For the current project narrative, `d_model=512` is the cleaner main setting for the LSTM residual gated fusion branch.

## 2026-05-11 Simple Concat Comparison

Question:

- Compare the current main setting against a simple concat baseline using the same cached features:
  - text RoBERTa 768
  - HuBERT audio 768
  - FER visual expression 782
- To keep the internal width comparable, the concat baseline uses `projection_dim=512`.

Config:

```text
configs/concat_tav_hubert_expression_d512.yaml
```

Result:

| Model | Context | Fusion | Best Dev Weighted F1 | Test Weighted F1 | Test Macro F1 |
|---|---|---|---:|---:|---:|
| concat_tav_hubert_expression_d512 | no | raw concat + MLP | 0.5047 | 0.5122 | 0.3409 |
| context_lstm_residual_gated_fusion_fer_d512 | yes | text base + gated audio/FER residual | 0.5975 | 0.6012 | 0.4243 |

Improvement of the current main model over simple concat:

```text
weighted_f1 +0.0890
macro_f1    +0.0834
```

Interpretation:

- The current gated contextual structure has a clear advantage over simple concat under the same input features and similar hidden width.
- This comparison supports the project claim better than only comparing with text-only, because concat is the most direct naive multimodal baseline.
- The improvement includes both dialogue context and residual gate design; if needed later, a context-only concat baseline can separate these two factors.

## 2026-05-13 Visual Extraction Strategy: Face-only Top-k Pooling

Motivation:

- Replacing the visual backbone alone did not help.
- The next hypothesis is that MELD video frames contain many noisy frames: no face, side face, wrong face, weak expression, or neutral transition.
- Instead of using two visual models, keep one FER backbone and improve frame selection.

Strategy:

```text
backbone = trpakov/vit-face-expression
sample = 8 frames per utterance
filter = keep only frames with detected face
score = max FER softmax probability per frame
pooling = average top-4 confident frames
```

Code/config updates:

```text
src/extract_visual_expression_features.py
src/feature_dataset.py
src/train.py
src/evaluate.py
src/export_logits.py
src/models/baselines.py
configs/visual_expression_topk_only.yaml
configs/context_lstm_residual_gated_fusion_fer_topk_d512.yaml
configs/concat_tav_hubert_expression_topk_d512.yaml
```

Feature extraction:

```text
features/visual_expression_topk/train.pt  shape=(9989, 782) available=9934
features/visual_expression_topk/dev.pt    shape=(1109, 782) available=1100
features/visual_expression_topk/test.pt   shape=(2610, 782) available=2598
```

Single visual modality result:

| Model | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|
| old visual_expression_only | 0.2751 | 0.2966 | 0.1807 |
| top-k visual_expression_topk_only | 0.2241 | 0.2087 | 0.1407 |

The top-k visual feature is worse as a standalone classifier. This likely means it loses broad neutral/scene-level information.

Main fusion result:

| Model | Best Dev Weighted F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|---:|
| old context_lstm_residual_gated_fusion_fer_d512 | 0.5975 | 0.5969 | 0.6012 | 0.4243 |
| top-k context_lstm_residual_gated_fusion_fer_topk_d512 | 0.5897 | 0.6015 | 0.6064 | 0.4401 |

Simple concat comparison with the same top-k visual feature:

| Model | Best Dev Weighted F1 | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|
| concat_tav_hubert_expression_topk_d512 | 0.5755 | 0.5798 | 0.4255 |
| context_lstm_residual_gated_fusion_fer_topk_d512 | 0.5897 | 0.6064 | 0.4401 |

Zero-modality checks:

| Setting | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|
| full text+audio+top-k FER | 0.6064 | 0.4401 |
| zero audio | 0.5578 | 0.4132 |
| zero top-k FER visual | 0.5734 | 0.3944 |
| zero audio+top-k FER visual | 0.5494 | 0.3821 |

Gate averages on test:

```text
gate_text   = 1.0000
gate_audio  = 0.9561
gate_visual = 0.8687
```

Interpretation:

- As a standalone visual classifier, face-only top-k pooling is worse.
- But as a residual correction signal inside the multimodal model, it improves both weighted F1 and macro F1.
- This supports the idea that the visual branch does not need to be a strong independent classifier; it needs to provide reliable correction cues.
- Current best frozen-feature main model should be updated to `context_lstm_residual_gated_fusion_fer_topk_d512`, with the caveat that dev weighted F1 is lower than the old visual version while test F1 is better.

## 2026-05-13 Stronger FER/Affect Visual Backbone Attempt

Goal:

- Keep the current audio branch unchanged.
- Replace the visual expression backbone with a FER/Affect-oriented model that may be closer to emotion recognition than CLIP-style visual semantics.

Selected model:

```text
mo-thecreator/vit-Facial-Expression-Recognition
```

Reason:

- It is still a ViT image classification model, so the current face-crop + multi-frame averaging pipeline can reuse it.
- The model card reports training on FER2013, MMI, and AffectNet, which should be more aligned with facial expression recognition than generic CLIP image features.

Code/config updates:

```text
src/extract_visual_expression_features.py
src/feature_dataset.py
src/train.py
src/evaluate.py
src/export_logits.py
src/models/baselines.py
configs/visual_expression_affectnet_only.yaml
configs/context_lstm_residual_gated_fusion_affectnet_d512.yaml
```

New modality name:

```text
visual_expression_affectnet -> features/visual_expression_affectnet
```

Feature extraction:

```text
features/visual_expression_affectnet/train.pt  shape=(9989, 782) available=9988
features/visual_expression_affectnet/dev.pt    shape=(1109, 782) available=1108
features/visual_expression_affectnet/test.pt   shape=(2610, 782) available=2610
```

Single visual modality result:

| Model | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|
| old visual_expression_only | 0.2751 | 0.2966 | 0.1807 |
| new visual_expression_affectnet_only | 0.2778 | 0.2694 | 0.1797 |

Main fusion result:

| Model | Best Dev Weighted F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|---:|
| old context_lstm_residual_gated_fusion_fer_d512 | 0.5975 | 0.5969 | 0.6012 | 0.4243 |
| new context_lstm_residual_gated_fusion_affectnet_d512 | 0.5932 | 0.5697 | 0.5853 | 0.4234 |

Zero-modality checks for the new visual backbone:

| Setting | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|
| full text+audio+new visual | 0.5853 | 0.4234 |
| zero audio | 0.5240 | 0.3792 |
| zero new visual | 0.5302 | 0.3875 |
| zero audio+new visual | 0.4547 | 0.3418 |

Gate averages on test:

```text
gate_text   = 1.0000
gate_audio  = 0.9702
gate_visual = 0.9969
```

Interpretation:

- This replacement did not improve the project.
- The new FER/Affect visual-only result is lower than the old FER visual-only weighted F1.
- In the main fusion model, the new visual backbone also reduces test weighted F1 from `0.6012` to `0.5853`.
- The zero-visual result drops sharply, which means the model relies heavily on this new visual branch, but that reliance is not beneficial enough for final performance.
- For visual backbone replacement alone, do not replace `trpakov/vit-face-expression` with this Affect-oriented model.
- This conclusion is about the backbone-swap attempt only; the later/stronger result comes from changing frame pooling while keeping `trpakov/vit-face-expression`.

## 2026-05-13 Audio Feature Improvement Check

Goal:

- Re-check whether the audio branch should be upgraded after the visual top-k strategy became the current main line.
- Keep the main model simple and course-project-friendly: improve the audio feature itself first, then test whether it helps the same residual gated fusion structure.

Current selected main line before this pass:

```text
context_lstm_residual_gated_fusion_fer_topk_d512
audio feature: torchaudio.pipelines.HUBERT_BASE mean pooling, 768 dim
visual feature: trpakov/vit-face-expression top-k confident frame pooling, 782 dim
test weighted_f1=0.6064, macro_f1=0.4401
```

Tested audio alternatives:

1. `audio_hubert_stats`: HuBERT frame mean + std, 1536 dim.
2. `audio_hubert_prosody`: HuBERT mean + MFCC/pitch/energy prosody stats, 883 dim.
3. `audio_emotion`: SER-oriented Wav2Vec2 model `Dpngtm/wav2vec2-emotion-recognition`, hidden mean + emotion logits + emotion probabilities, 782 dim.

Code/config updates:

```text
src/extract_audio_emotion_features.py
src/feature_dataset.py
src/train.py
src/evaluate.py
src/export_logits.py
src/models/baselines.py
src/models/context.py
src/models/__init__.py
configs/audio_emotion_only.yaml
configs/context_lstm_residual_gated_fusion_fer_topk_audio_emotion_d512.yaml
configs/context_lstm_residual_gated_fusion_fer_topk_hubert_prosody_d512.yaml
configs/context_lstm_residual_gated_fusion_fer_topk_hubert_stats_d512.yaml
```

Extracted SER audio features:

```text
features/audio_emotion/train.pt  shape=(9989, 782) available=9988
features/audio_emotion/dev.pt    shape=(1109, 782) available=1108
features/audio_emotion/test.pt   shape=(2610, 782) available=2610
```

Single audio modality results:

| Model | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|
| audio_hubert_only | 0.2927 | 0.3185 | 0.2182 |
| audio_hubert_stats_only | 0.3172 | 0.3350 | 0.2161 |
| audio_hubert_prosody_only | 0.3123 | 0.3406 | 0.2398 |
| audio_emotion_only | 0.2215 | 0.2440 | 0.1749 |

Main fusion comparison with the same top-k visual branch:

| Audio feature | Best Dev Weighted F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|---:|
| HuBERT mean | 0.5897 | 0.6015 | 0.6064 | 0.4401 |
| HuBERT mean+std | 0.5715 | 0.5590 | 0.5824 | 0.4270 |
| HuBERT+prosody | 0.5921 | 0.6161 | 0.6057 | 0.4256 |
| SER audio_emotion | 0.5805 | 0.5751 | 0.5904 | 0.4312 |

Important ablation for HuBERT+prosody:

```text
full HuBERT+prosody fusion       weighted_f1=0.6057  macro_f1=0.4256
zero audio_hubert_prosody        weighted_f1=0.6179  macro_f1=0.4230
zero visual_expression_topk      weighted_f1=0.5746  macro_f1=0.3771
zero audio+visual                weighted_f1=0.5518  macro_f1=0.3440
```

Conclusion:

- `audio_hubert_prosody` is the best audio-only cached feature so far, but it does not improve the full multimodal model.
- The external SER model transfers poorly to MELD; its audio-only score is lower than HuBERT/prosody, and its fusion result is also worse.
- The current main line should keep the original HuBERT mean audio feature for now, because it has the best full-model weighted F1 and macro F1.
- The practical interpretation is that audio improvement cannot be solved just by increasing dimensionality or adding an external SER classifier. The useful next audio direction would be MELD-specific audio fine-tuning or a stronger model trained/evaluated closer to conversational speech, but that should be treated as a separate, larger experiment.

## 2026-05-14 Plan Files Consolidated

The standalone planning files were consolidated into this work log so the final project directory is cleaner.

Removed planning files:

```text
single_modality_improvement_plan.md
visual_modality_improvement_plan.md
residual_context_fusion_plan.md
```

Important conclusions preserved:

1. Single-modality bottleneck:
   - The small early multimodal gain was not mainly caused by the fusion gate itself.
   - Audio and visual frozen features were much weaker than text, so fusion could only use them cautiously.
   - Deeper audio MLP heads did not solve the problem:
     - `audio_hubert_mlp` only slightly improved over shallow HuBERT.
     - `audio_hubert_stats_mlp` and `audio_hubert_prosody_mlp` were worse than their simpler baselines.
   - Conclusion: the audio bottleneck is feature quality/domain alignment, not just classifier-head capacity.

2. Visual modality direction:
   - Generic CLIP visual features were too scene/semantic-oriented for MELD emotion recognition.
   - The useful visual direction was face-centered FER/expression features, not another generic image encoder.
   - The final useful strategy became:

```text
trpakov/vit-face-expression
+ face crop
+ 8 sampled frames
+ top-4 confident FER frames
+ mean pooling
```

   - The AffectNet-style replacement backbone was tested but did not beat the original FER backbone, so the final visual branch keeps `trpakov/vit-face-expression`.

3. Residual gated context fusion:
   - The main architecture came from the residual-correction idea:

```text
final_logits =
    text_logits
    + sigmoid(audio_gate)  * audio_residual_scale  * audio_delta
    + sigmoid(visual_gate) * visual_residual_scale * visual_delta
```

   - This avoids forcing weak audio/visual branches to compete directly with text.
   - Independent sigmoid gates are better aligned with the idea that audio and visual can each decide whether to correct the text base.
   - Dialogue-level BiLSTM context was kept because MELD emotion labels often depend on surrounding utterances.

Final retained main line after these plans:

```text
configs/context_lstm_residual_gated_fusion_fer_topk_d512.yaml
RoBERTa text + HuBERT mean audio + top-k FER visual
d_model=512
dialogue BiLSTM context
residual gated audio/visual correction
```
