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
