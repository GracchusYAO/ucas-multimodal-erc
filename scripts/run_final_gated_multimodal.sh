#!/usr/bin/env bash
set -euo pipefail

# 复现当前主结果：强文本 ensemble + top-3 多模态辅助 + class-aware gate。
# 运行前需要先用 src.export_logits.py 导出对应 dev/test logits。

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-results/offline_gated_multimodal_class_top3}"

"${PYTHON_BIN}" -u -m src.evaluate_offline_gated_fusion \
  --text-logit-dir results/logits/text_finetune_context \
  --text-logit-dir results/logits/text_finetune_context_unweighted \
  --text-logit-dir results/logits/text_finetune_context5_unweighted \
  --text-logit-dir results/logits/text_finetune_context8_unweighted \
  --text-logit-dir results/logits/text_finetune_roberta_large_context5_unweighted \
  --text-logit-dir results/logits/text_finetune_deberta_v3_base_context5_unweighted \
  --aux-logit-dir results/logits/dgf_dropout \
  --aux-logit-dir results/logits/late_fusion_hubert \
  --aux-logit-dir results/logits/quality_late_fusion_hubert \
  --split test \
  --output-dir "${OUTPUT_DIR}" \
  --seed 114514 \
  --text-average logit \
  --aux-average prob \
  --text-weight-search-trials 3000 \
  --aux-weight-search-trials 1000 \
  --include-zero-alpha \
  --gate-mode class \
  --class-gate-rounds 2
