# TODO: Multimodal Emotion Recognition in Conversations

Project goal: build a course-project-level multimodal emotion recognition system on MELD, focusing on model design rather than heavy raw-data processing.

Core version: **Dynamic Reliability-Gated Fusion + Modality Dropout + BiGRU Context Encoder**

---

## 0. Project Setup

- [ ] Confirm repository structure:

```text
data/
features/
src/
  models/
configs/
results/
notebooks/
README.md
.gitignore
```

- [ ] Add `data/`, `features/`, `results/`, `checkpoints/`, `runs/`, `wandb/` to `.gitignore`.
- [ ] Add basic dependencies to `requirements.txt`:
  - `torch`
  - `transformers`
  - `torchaudio`
  - `librosa`
  - `opencv-python`
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `tqdm`
  - `matplotlib`
  - `seaborn`
  - `pyyaml`

---

## 1. Dataset Preparation

Dataset: **MELD**

- [x] Place MELD files under:

```text
data/meld/
```

Expected metadata files:

```text
train_sent_emo.csv
dev_sent_emo.csv
test_sent_emo.csv
```

Expected media files:

```text
train_splits/
dev_splits_complete/
output_repeated_splits_test/
```

- [x] Implement `src/dataset.py`.
- [x] Parse each utterance with fields:
  - `Dialogue_ID`
  - `Utterance_ID`
  - `Utterance`
  - `Emotion`
  - `Sentiment`
  - media path
- [x] Keep official train/dev/test split.
- [x] Map 7 emotion labels to integer ids:

```python
emotion2id = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}
```

---

## 2. Feature Extraction

Use frozen pretrained encoders. Save extracted features to `.npy` or `.pt`.

### 2.1 Text Feature Extraction

Model: `FacebookAI/roberta-base`

- [x] Create `src/extract_text_features.py`.
- [x] Input: utterance text.
- [x] Output: utterance-level text embedding.

Default settings:

```yaml
text_model: FacebookAI/roberta-base
text_pooling: mean        # [TUNE] options: mean, cls
max_length: 128           # [TUNE] options: 64, 128
batch_size_text: 32       # [TUNE] adjust by GPU memory
output_dim_text: 768
```

Output path:

```text
features/text_roberta/{split}.pt
```

---

### 2.2 Audio Feature Extraction

Model: `facebook/wav2vec2-base`

- [x] Create `src/extract_audio_features.py`.
- [x] Convert audio to 16 kHz.
- [x] Input: utterance-level audio clip.
- [x] Output: mean-pooled Wav2Vec2 hidden states.

Default settings:

```yaml
audio_model: facebook/wav2vec2-base
sample_rate: 16000
audio_pooling: mean       # [TUNE] options: mean, attentive
batch_size_audio: 4       # [TUNE] options: 2, 4, 8
max_audio_seconds: 12     # [TUNE] truncate long clips if needed
output_dim_audio: 768
```

Output path:

```text
features/audio_wav2vec2/{split}.pt
```

---

### 2.3 Visual Feature Extraction

Model: `openai/clip-vit-base-patch32`

- [x] Create `src/extract_visual_features.py`.
- [x] Sample frames from each utterance video.
- [x] Extract CLIP image features for each frame.
- [x] Mean-pool frame features into one utterance-level visual embedding.

Default settings:

```yaml
visual_model: openai/clip-vit-base-patch32
num_frames: 4             # [TUNE] options: 4, 8
frame_sampling: uniform
batch_size_visual: 16     # [TUNE] adjust by GPU memory
output_dim_visual: 512
```

Output path:

```text
features/visual_clip/{split}.pt
```

---

## 3. Baseline Models

Create `src/models/baselines.py`.

### 3.1 Text-only Baseline

- [x] Model: text feature -> projection MLP -> classifier.
- [x] Use this as the strongest simple baseline.

### 3.2 Trimodal Concatenation Baseline

- [x] Model: concatenate text/audio/visual features -> MLP classifier.
- [x] Compare this against dynamic gated fusion.

Default baseline settings:

```yaml
projection_dim: 256       # [TUNE] options: 128, 256, 512
dropout: 0.3              # [TUNE] options: 0.1, 0.3, 0.5
num_classes: 7
```

---

## 4. Main Model: Dynamic Reliability-Gated Fusion

Create `src/models/fusion.py`.

### 4.1 Modality Projection

Project all modalities into the same dimension:

```text
text:   768 -> d_model
audio:  768 -> d_model
visual: 512 -> d_model
```

Default settings:

```yaml
d_model: 256              # [TUNE] options: 128, 256, 512
projection_activation: relu
projection_dropout: 0.3   # [TUNE]
use_layernorm: true       # [TUNE]
```

### 4.2 Dynamic Modality Gate

Implement:

```python
gate = softmax(MLP(concat([z_text, z_audio, z_visual])))
z_fused = gate_text * z_text + gate_audio * z_audio + gate_visual * z_visual
```

- [x] Return both `z_fused` and `gate_weights`.
- [x] Save gate weights during evaluation for visualization.

Default settings:

```yaml
gate_hidden_dim: 128      # [TUNE] options: 64, 128, 256
gate_dropout: 0.2         # [TUNE]
```

---

## 5. Modality Dropout

Implement modality dropout inside the main model.

- [x] During training only, randomly mask one or more modalities.
- [x] Do not apply modality dropout during validation/test.
- [x] Start with moderate dropout to avoid hurting convergence.

Default settings:

```yaml
use_modality_dropout: true
modality_dropout_p: 0.3   # [TUNE] options: 0.1, 0.2, 0.3, 0.5
drop_text_p: 0.1          # [TUNE]
drop_audio_p: 0.2         # [TUNE]
drop_visual_p: 0.2        # [TUNE]
```

Implementation note:

```python
if training:
    randomly set selected modality feature to zero
```

---

## 6. Dialogue Context Encoder

Create `src/models/context.py`.

Use a 1-layer BiGRU over utterances within the same dialogue.

Input:

```text
[z_fused_1, z_fused_2, ..., z_fused_n]
```

Output:

```text
context-aware utterance representations
```

Default settings:

```yaml
context_encoder: bigru
context_hidden_dim: 256   # [TUNE] options: 128, 256
context_num_layers: 1     # [TUNE] keep 1 for first version
bidirectional: true
context_dropout: 0.3      # [TUNE]
```

Classifier:

```yaml
classifier_hidden_dim: 256  # [TUNE]
classifier_dropout: 0.3     # [TUNE]
```

---

## 7. Training Pipeline

Create `src/train.py`.

- [x] Load cached text/audio/visual features.
- [x] Load labels and dialogue ids.
- [x] Batch by dialogue if using BiGRU context.
- [x] Support configs from YAML.
- [x] Use class-weighted cross entropy.

Default settings:

```yaml
optimizer: adamw
learning_rate: 1e-4       # [TUNE] options: 5e-5, 1e-4, 3e-4
weight_decay: 1e-4        # [TUNE]
batch_size_dialogue: 8    # [TUNE] number of dialogues per batch
max_epochs: 50            # [TUNE]
early_stopping_patience: 5
loss: weighted_cross_entropy
seed: 114514
```

- [x] Save best checkpoint by validation weighted F1.

Output:

```text
checkpoints/best_model.pt
results/train_log.csv
```

---

## 8. Evaluation

Create `src/evaluate.py`.

Metrics:

- [x] Accuracy
- [x] Weighted F1
- [x] Macro F1
- [x] Per-class F1
- [x] Confusion matrix

Primary metrics:

```text
Weighted F1
Macro F1
```

Output:

```text
results/metrics.json
results/confusion_matrix.png
```

---

## 9. Required Experiments

Run these experiments in order.

### Stage A: Must-have baselines

- [x] E1: Text-only
- [x] E2: Text + Audio + Visual concat
- [x] E3: Dynamic Gated Fusion without context
- [x] E4: Dynamic Gated Fusion + Modality Dropout
- [x] E5: Full model = Dynamic Gated Fusion + Modality Dropout + BiGRU Context

### Stage B: Optional but recommended

- [x] E6: Audio-only
- [x] E7: Visual-only
- [x] E8: Text + Audio
- [x] E9: Text + Visual
- [ ] E10: Full model without modality dropout
- [ ] E11: Full model without BiGRU context

### Stage C: Stronger Text and Multimodal Extensions

- [x] Fine-tune a text encoder instead of only training on frozen RoBERTa features.
- [x] Add speaker-aware dialogue context text input.
- [x] Add text fine-tune ensemble evaluation.
- [x] Add frozen-feature audio-only and visual-only baselines.
- [x] Rebuild audio features with a stronger speech encoder such as Whisper or HuBERT.
- [x] Rebuild visual features with face-centered frames or an expression-focused visual encoder.
- [x] Add audio-only strong-feature baseline before fusing again.
- [x] Add visual-only fine-tuned/strong-feature baseline before fusing again.
- [x] Add simple concat baseline with stronger audio and face-centered visual features.
- [x] Add late-fusion logits baseline between strong text/audio/visual models.
- [x] Add quality-aware multimodal gating using missing-modality and modality-confidence signals.
- [ ] Split strong text logits and multimodal logits into offline files before mixed ensemble.

---

## 10. Visualization and Analysis

Create `src/visualize.py`.

- [x] Plot F1 comparison across experiments.
- [x] Plot confusion matrix.
- [x] Plot average modality gate weights by emotion class.
- [x] Plot performance under missing modality settings:
  - no text
  - no audio
  - no visual
  - no audio + no visual

Expected outputs:

```text
results/f1_comparison.png
results/confusion_matrix.png
results/gate_weights_by_emotion.png
results/missing_modality_analysis.png
```

---

## 11. Suggested Config Files

Create:

```text
configs/text_only.yaml
configs/concat_tav.yaml
configs/dgf.yaml
configs/dgf_dropout.yaml
configs/dgf_context.yaml
```

Main recommended config:

```yaml
model_name: dgf_context
d_model: 256
dropout: 0.3

use_modality_dropout: true
modality_dropout_p: 0.3
drop_text_p: 0.1
drop_audio_p: 0.2
drop_visual_p: 0.2

context_encoder: bigru
context_hidden_dim: 256
context_num_layers: 1
bidirectional: true

optimizer: adamw
learning_rate: 1e-4
weight_decay: 1e-4
batch_size_dialogue: 8
max_epochs: 50
early_stopping_patience: 5
seed: 114514
```

---

## 12. Implementation Priority

Recommended order for Codex:

1. [ ] Build dataset parser.
2. [ ] Implement text feature extraction.
3. [ ] Implement audio feature extraction.
4. [ ] Implement visual feature extraction.
5. [ ] Implement Text-only baseline.
6. [ ] Implement T+A+V concat baseline.
7. [ ] Implement Dynamic Reliability-Gated Fusion.
8. [ ] Add Modality Dropout.
9. [ ] Add BiGRU Dialogue Context Encoder.
10. [ ] Implement training and evaluation scripts.
11. [ ] Add experiment configs.
12. [ ] Add visualization scripts.
13. [ ] Update README with usage instructions.

---

## 13. Notes for Scope Control

- The first version used frozen features and DGF to establish the project pipeline.
- The stronger version may fine-tune text encoders and rebuild audio/visual features.
- Do not claim multimodal improvement unless audio-only or visual-only baselines show useful signal.
- If frozen CLIP/Wav2Vec2 features remain noisy, prefer late fusion and quality-aware gating over blindly adding modalities.
- If time remains, compare CLIP visual features with face-centered or VideoMAE-style visual features as an optional extension.
