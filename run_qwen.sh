#!/usr/bin/env bash
# ============================================
# RSN Regeneration Experiments (Mistral & Qwen)
# Author: Paveen Huang
# Date: 2025-07-28
# ============================================

set -e  # Exit on error
set -u  # Treat unset vars as errors

# ---------- QWEN3 8B ----------
MODEL_QWEN="qwen3"
MODEL_DIR_QWEN="Qwen/Qwen3-8B"
HS_QWEN="qwen3"
SIZE_QWEN="8B"

echo "===== Running Qwen3-8B Experiments ====="

# α=3 NMD 0.5%
python get_answer_regenerate_logits.py \
  --data data1 \
  --model ${MODEL_QWEN} \
  --model_dir ${MODEL_DIR_QWEN} \
  --hs ${HS_QWEN} \
  --size ${SIZE_QWEN} \
  --type non \
  --percentage 0.5 \
  --configs 3-17-26 \
  --mask_type nmd \
  --ans_file answer_mdf_mmlu \
  --tail_len 1 \
  --suite default \
  --E

# NMD/FVs 100%
python get_answer_regenerate_logits.py \
  --data data1 \
  --model ${MODEL_QWEN} \
  --model_dir ${MODEL_DIR_QWEN} \
  --hs ${HS_QWEN} \
  --size ${SIZE_QWEN} \
  --type non \
  --percentage 100 \
  --configs 4-25-26 1-36-37 \
  --mask_type nmd \
  --ans_file answer_mdf_mmlu \
  --tail_len 1 \
  --suite default \
  --E

# α=1 T-Test
python get_answer_regenerate_logits.py \
  --data data1 \
  --model ${MODEL_QWEN} \
  --model_dir ${MODEL_DIR_QWEN} \
  --hs ${HS_QWEN} \
  --size ${SIZE_QWEN} \
  --type non \
  --percentage 0.5 \
  --configs 1-1-37 \
  --mask_type ttest \
  --ans_file answer_ttest_mmlu \
  --tail_len 1 \
  --suite default \
  --E

echo "===== All experiments completed successfully Qwen ====="