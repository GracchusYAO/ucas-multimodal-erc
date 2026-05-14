#!/usr/bin/env bash
set -euo pipefail

# 复现当前最终主线：
# RoBERTa text + HuBERT audio + top-k FER visual
# dialogue BiLSTM + residual gated fusion.
#
# 运行前需要已经准备好缓存特征：
#   features/text_roberta/{train,dev,test}.pt
#   features/audio_hubert/{train,dev,test}.pt
#   features/visual_expression_topk/{train,dev,test}.pt

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-configs/context_lstm_residual_gated_fusion_fer_topk_d512.yaml}"
RUN_NAME="${RUN_NAME:-context_lstm_residual_gated_fusion_fer_topk_d512}"

OUTPUT_DIR="${OUTPUT_DIR:-results/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-logs}"
EVAL_ROOT="${EVAL_ROOT:-results/evaluate}"
CHECKPOINT="${CHECKPOINT_DIR}/best_context_lstm_residual_gated_fusion.pt"

mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" -u -m src.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  2>&1 | tee "${LOG_DIR}/train_${RUN_NAME}.log"

"${PYTHON_BIN}" -u -m src.evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --device "${DEVICE}" \
  --output-dir "${EVAL_ROOT}/${RUN_NAME}_test" \
  2>&1 | tee "${LOG_DIR}/evaluate_${RUN_NAME}_test.log"

"${PYTHON_BIN}" -u -m src.evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --device "${DEVICE}" \
  --zero-modality audio_hubert \
  --output-dir "${EVAL_ROOT}/${RUN_NAME}_zero_audio_test" \
  2>&1 | tee "${LOG_DIR}/evaluate_${RUN_NAME}_zero_audio_test.log"

"${PYTHON_BIN}" -u -m src.evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --device "${DEVICE}" \
  --zero-modality visual_expression_topk \
  --output-dir "${EVAL_ROOT}/${RUN_NAME}_zero_visual_test" \
  2>&1 | tee "${LOG_DIR}/evaluate_${RUN_NAME}_zero_visual_test.log"

"${PYTHON_BIN}" -u -m src.evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --device "${DEVICE}" \
  --zero-modality audio_hubert \
  --zero-modality visual_expression_topk \
  --output-dir "${EVAL_ROOT}/${RUN_NAME}_zero_audio_visual_test" \
  2>&1 | tee "${LOG_DIR}/evaluate_${RUN_NAME}_zero_audio_visual_test.log"
