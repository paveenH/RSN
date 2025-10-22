#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 16:33:54 2025

@author: paveenhuang
"""

#!/bin/bash
# run_all.sh

MODEL="mistral"
MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.3"
SIZE="7B"
TYPE="non"
SUITE="default"

DATASETS=(
  "medqa/medqa_source_test.json"
  "pubmedqa/pubmedqa_labeled_train.json"
  "factor/factor_mc.json"
  "gpqa/gpqa_train.json"
  "arlsat/arlsat_all.json"
  "logiqa/logiqa_mrc.json"
)

for FILE in "${DATASETS[@]}"; do
  NAME=$(echo "$FILE" | cut -d'/' -f1)

  echo "=== Running $NAME ==="
  python get_answer_logits_mmlupro.py \
    --data data2 \
    --model "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --type "$TYPE" \
    --test_file "$FILE" \
    --ans_file "answer/answer_orig_${NAME}_cot" \
    --suite "$SUITE" \
    --cot
done

echo "===== Running Mistral-7B Experiments ====="

# α=3 NMD 0.5%
python get_answer_regenerate_logits.py \
  --data data2 \
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
  --data data2 \
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
  --data data2 \
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