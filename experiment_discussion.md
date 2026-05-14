# Multimodal ERC 实验讨论稿

这份文档记录当前最终主线配置和对应实验结果。目标是方便组内讨论和后续写报告，因此只保留我们准备作为主线汇报的模型、消融实验，以及它相对单模态和简单融合 baseline 的提升。

## 1. 最终主线模型架构

当前建议作为最终主线的配置文件是：

```text
configs/context_lstm_residual_gated_fusion_fer_topk_d512.yaml
```

模型名称是：

```text
context_lstm_residual_gated_fusion_fer_topk_d512
```

它使用 MELD 的三种模态：

| 模态 | 当前基座模型 / 特征来源 | 原始特征维度 |
|---|---|---:|
| 文本 | `FacebookAI/roberta-base`，utterance-level mean pooling | 768 |
| 音频 | `torchaudio.pipelines.HUBERT_BASE`，utterance-level mean pooling | 768 |
| 视觉 | `trpakov/vit-face-expression`，face-only + top-k confident frame pooling | 782 |

其中视觉 FER 特征由三部分组成：

```text
ViT CLS embedding 768
+ emotion logits 7
+ emotion probabilities 7
= 782
```

当前视觉提取不是简单平均所有采样帧，而是：

```text
每个 utterance 均匀采样 8 帧
只保留检测到人脸的帧
用 FER softmax 最大概率作为帧置信度
选 top-4 confident frames
平均这几帧的 FER feature
```

这样做的动机是：影视对话视频里有大量侧脸、无脸、旁人脸、弱表情或中性过渡帧。直接平均所有帧会把有效表情信号稀释掉；top-k pooling 让视觉分支更偏向“更稳定、更像表情帧”的片段。

### 1.1 内部表示维度

三路模态进入融合模型后，先分别投影到：

```text
d_model = 512
```

选择 `512` 的原因是：原始基座特征基本都在 768 维左右，之前统一压到 `256` 可能过度压缩音频和视觉信息；但直接提升到 `768` 后训练不稳定、泛化没有变好。因此当前主线固定为 `512`。

### 1.2 融合结构

模型不是简单拼接三种模态，而是采用：

```text
文本主判断 + 音频/视觉 residual correction + dialogue-level BiLSTM context
```

整体流程：

```text
text / HuBERT audio / FER visual
        ↓
各自线性投影到 512 维
        ↓
按 dialogue 顺序输入 BiLSTM，得到上下文表示
        ↓
text_head 产生 text_logits，作为主预测
audio_delta_head 产生 audio_delta，只负责修正文本预测
visual_delta_head 产生 visual_delta，只负责修正文本预测
        ↓
sigmoid gate 分别控制 audio_delta 和 visual_delta 的修正强度
        ↓
final_logits
```

核心公式：

```text
final_logits =
    text_logits
    + sigmoid(audio_gate)  * audio_residual_scale  * audio_delta
    + sigmoid(visual_gate) * visual_residual_scale * visual_delta
```

当前设置：

```text
audio_residual_scale  = 1.0
visual_residual_scale = 0.4
audio_gate_bias       = -1.0
visual_gate_bias      = -2.0
```

这样设计的原因：

- 文本是 MELD 上最可靠的模态，因此文本作为基座。
- 音频和视觉不是直接与文本抢最终分类权重，而是学习“在上下文中如何修正文本 logits”。
- audio gate 和 visual gate 是独立 sigmoid，不是 softmax，因此音频和视觉不需要互相竞争。
- 使用 BiLSTM 是为了让模型看到同一 dialogue 中前后 utterance 的情绪和语境变化。

## 2. 当前主线对应的单模态基线

为了说明多模态融合是否有意义，先看当前三种基座特征各自单独分类的效果。

| Model | Input Feature | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---|---:|---:|---:|
| `text_only` | RoBERTa text | 0.5487 | 0.5722 | 0.4158 |
| `audio_hubert_only` | HuBERT audio | 0.2927 | 0.3185 | 0.2182 |
| `visual_expression_topk_only` | FER top-k visual expression | 0.2241 | 0.2087 | 0.1407 |
| **最终主线** | Text + HuBERT + FER top-k | **0.6015** | **0.6064** | **0.4401** |

可以看到，文本仍然是最强的单模态。音频和视觉单独分类明显较弱，尤其 top-k 视觉单独分类并不好。因此当前模型没有采用“多模态平均”或“直接让弱模态与文本竞争”的思路，而是让文本作为主预测，再让音频和视觉在上下文中做 residual correction。

最终主线相对各个单模态的提升如下：

| 对比对象 | Weighted F1 提升 | Macro F1 提升 |
|---|---:|---:|
| 相比 `text_only` | +0.0342 | +0.0243 |
| 相比 `audio_hubert_only` | +0.2879 | +0.2219 |
| 相比 `visual_expression_topk_only` | +0.3977 | +0.2994 |

其中最重要的是与 `text_only` 的对比，因为文本是 MELD 中最强、最可靠的单模态。最终主线相比 text-only 的 test weighted F1 从 `0.5722` 提升到 `0.6064`，说明多模态结构确实带来了增益。

这里有一个重要现象：top-k 视觉单模态比原始全帧平均视觉更弱，但放进 residual gated fusion 后反而更好。说明它不适合作为独立分类器，却可能提供了更干净的“修正信号”。

这也解释了为什么后面的消融里，`zero audio+visual` 不能简单等同于独立训练的 `text_only`。它是在已经训练好的多模态结构里遮蔽非文本模态，用来观察模型对音频和视觉的依赖程度。

### 2.1 音频增强尝试后的取舍

我们额外测试了几种音频特征，目的是确认当前主线是否应该替换 HuBERT mean audio。

| Audio Feature | Audio-only Test Weighted F1 | Fusion Test Weighted F1 | Fusion Test Macro F1 |
|---|---:|---:|---:|
| HuBERT mean | 0.3185 | 0.6064 | 0.4401 |
| HuBERT mean+std | 0.3350 | 0.5824 | 0.4270 |
| HuBERT+prosody | 0.3406 | 0.6057 | 0.4256 |
| SER audio emotion | 0.2440 | 0.5904 | 0.4312 |

结论是：`HuBERT+prosody` 的单模态音频最强，但放进当前融合结构后没有超过 HuBERT mean 主线。外部 SER 模型 `Dpngtm/wav2vec2-emotion-recognition` 迁移到 MELD 后表现更弱，说明“换成情感音频模型”不一定自动有效，域差异和标签差异会很明显。

另一个关键现象是：`HuBERT+prosody` 融合模型在 test set 上 full model weighted F1 为 `0.6057`，但 zero audio 后反而达到 `0.6179`。这说明它虽然提升了 audio-only 分支，但在当前融合结构里并不稳定，不能作为最终主线。

因此当前主线仍然保留：

```text
audio = HuBERT mean pooling, 768 dim
```

这不是因为 prosody 或 SER 没有价值，而是因为课程项目当前需要一个能被消融实验支撑的主线结果。现阶段最稳的证据仍然来自 HuBERT mean + top-k FER visual + LSTM residual gated fusion。

## 3. 当前主线实验结果

训练配置：

```text
seed = 114514
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 1e-4
loss = weighted cross entropy
batch_size_dialogue = 8
early_stopping_patience = 5
```

主线模型结果：

| Model | Best Dev Weighted F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|---:|
| `context_lstm_residual_gated_fusion_fer_topk_d512` | 0.5897 | 0.6015 | 0.6064 | 0.4401 |

目前可以把这个作为 frozen-feature 多模态主线结果。

## 4. 与简单 Concat 的对比

为了验证模型结构是否真的有价值，我们补了一个简单拼接 baseline：

```text
concat_tav_hubert_expression_topk_d512
```

它使用完全相同的三路输入特征：

```text
text_roberta 768
audio_hubert 768
visual_expression_topk 782
```

并且把 concat baseline 的隐藏层宽度也设为：

```text
projection_dim = 512
```

对比结果：

| Model | Context | Fusion | Best Dev Weighted F1 | Test Weighted F1 | Test Macro F1 |
|---|---|---|---:|---:|---:|
| simple concat top-k d512 | no | raw concat + MLP | 0.5755 | 0.5798 | 0.4255 |
| LSTM residual gate top-k d512 | yes | text base + gated audio/FER residual | 0.5897 | 0.6064 | 0.4401 |

主线模型相对简单 concat 的提升：

```text
Weighted F1: +0.0266
Macro F1:    +0.0146
```

这个结果说明：在相同输入特征和相近隐藏维度下，当前模型仍然优于简单拼接。top-k 视觉策略本身已经让 concat baseline 变强，但上下文建模和 residual gated fusion 仍然带来额外提升。

## 5. 主线模型的模态消融

为了确认音频和视觉是否真的被模型使用，我们在 test set 上做了 zero-modality 消融。

| Setting | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|
| full text+audio+FER top-k | 0.6015 | 0.6064 | 0.4401 |
| zero audio | 0.5398 | 0.5578 | 0.4132 |
| zero FER top-k visual | 0.5686 | 0.5734 | 0.3944 |
| zero audio+FER top-k visual | 0.5368 | 0.5494 | 0.3821 |

消融结果对应的下降幅度：

| 消融设置 | Weighted F1 下降 | Macro F1 下降 |
|---|---:|---:|
| 去掉 audio | -0.0486 | -0.0269 |
| 去掉 visual | -0.0330 | -0.0457 |
| 同时去掉 audio+visual | -0.0570 | -0.0580 |

可以得到三个结论：

1. 音频模态贡献明显。去掉 HuBERT audio 后，weighted F1 从 `0.6064` 降到 `0.5578`。
2. top-k 视觉模态贡献也更明显。去掉 FER top-k visual 后，weighted F1 从 `0.6064` 降到 `0.5734`。
3. 两个非文本模态整体有效。去掉 audio+visual 后，weighted F1 降到 `0.5494`，比完整模型低 `0.0570`。

这组消融对我们的项目叙事很重要：模型不是形式上用了多模态，而是在 test set 上确实依赖了非文本信息。

## 6. 当前讨论结论

目前可以对外讨论的主线结论是：

```text
我们采用 RoBERTa 文本特征、HuBERT 音频特征和 face-only top-k FER 表情视觉特征。
由于文本是最可靠的模态，最终模型以文本 logits 为基座，
再用 dialogue-level BiLSTM 建模上下文，
并通过独立 sigmoid gate 控制音频和视觉 residual 对文本预测的修正。
```

实验上：

```text
主线模型 test weighted F1 = 0.6064
text-only test weighted F1 = 0.5722
简单 concat test weighted F1 = 0.5798
相比 text-only 提升 = +0.0342
相比 simple concat 提升 = +0.0266
```

模态消融也支持这个设计：

```text
full model             0.6064
zero audio             0.5578
zero visual            0.5734
zero audio + visual    0.5494
```

因此目前比较稳妥的说法是：

```text
相比单模态文本和简单拼接融合，当前的上下文残差门控融合能更有效地利用音频和视觉信息；
zero-modality 消融表明音频和 top-k FER 视觉都对最终预测有明确贡献。
```

需要注意的是，这个结果还不是 SOTA 水平，也不是端到端 fine-tuning 后的最终上限。它目前的意义在于：作为课程项目中的 frozen-feature 多模态模型，已经能用实验说明“我们设计的融合结构比简单 concat 更有效”，并且消融实验可以支撑这一点。
