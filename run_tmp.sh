#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 18:12:30 2025

@author: paveenhuang
"""

#!/usr/bin/env bash
# ============================================
# RSN Regeneration Experiments (Mistral & Qwen)
# Author: Paveen Huang
# Date: 2025-07-28
# ============================================

set -e  # Exit on error
set -u  # Treat unset vars as errors

# ---------- MISTRAL 7B ----------
MODEL_MISTRAL="mistral"
MODEL_DIR_MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"
HS_MISTRAL="mistral"
SIZE_MISTRAL="7B"

echo "===== Running Mistral-7B Experiments ====="

# α=3 NMD 0.5%
python get_answer_regenerate_logits.py \
  --model ${MODEL_MISTRAL} \
  --model_dir ${MODEL_DIR_MISTRAL} \
  --hs ${HS_MISTRAL} \
  --size ${SIZE_MISTRAL} \
  --type non \
  --percentage 0.5 \
  --configs 3-14-22 \
  --mask_type nmd \
  --ans_file answer_mdf_mmlu \
  --tail_len 1 \
  --suite default \
  --E

# NMD/FVs 100%
python get_answer_regenerate_logits.py \
  --model ${MODEL_MISTRAL} \
  --model_dir ${MODEL_DIR_MISTRAL} \
  --hs ${HS_MISTRAL} \
  --size ${SIZE_MISTRAL} \
  --type non \
  --percentage 100 \
  --configs 4-21-22 1-32-33 \
  --mask_type nmd \
  --ans_file answer_mdf_mmlu \
  --tail_len 1 \
  --suite default \
  --E

# α=1 T-Test
python get_answer_regenerate_logits.py \
  --model ${MODEL_MISTRAL} \
  --model_dir ${MODEL_DIR_MISTRAL} \
  --hs ${HS_MISTRAL} \
  --size ${SIZE_MISTRAL} \
  --type non \
  --percentage 0.5 \
  --configs 1-1-33 \
  --mask_type ttest \
  --ans_file answer_ttest_mmlu \
  --tail_len 1 \
  --suite default \
  --E

echo "===== All experiments completed successfully Mistral====="