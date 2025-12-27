#!/bin/bash

# --- Configuration ---
DATA_DIR="data2"
MODEL_NAME="llama3_base"
MODEL_PATH="meta-llama/Llama-3.1-8B"
MODEL_SIZE="8B"
TYPE="non"
SUITE="default"
# Regenerate
HS_TYPE="llama3" 
PERCENTAGE="0.5"
CONFIGS="3-11-20 4-11-20"
MASK_TYPE="nmd"

echo "=================================================="
echo "Start..."
echo "=================================================="

# 1. Generate Logits (MMLU Original)
echo "[1/4] Running: Get Answer Logits (MMLU)..."
python get_answer_logits.py \
    --data "$DATA_DIR" \
    --model "$MODEL_NAME" \
    --model_dir "$MODEL_PATH" \
    --size "$MODEL_SIZE" \
    --type "$TYPE" \
    --suite "$SUITE" \
    --ans_file mmlu_orig

# 2. Generate Logits (MMLU-Pro)
echo "[2/4] Running: Get Answer Logits (MMLU-Pro)..."
python get_answer_logits_mmlupro.py \
    --data "$DATA_DIR" \
    --model "$MODEL_NAME" \
    --model_dir "$MODEL_PATH" \
    --size "$MODEL_SIZE" \
    --type "$TYPE" \
    --test_file mmlupro/mmlupro_test.json \
    --ans_file answer/answer_orig_mmlupro \
    --suite "$SUITE"

# 3. Regenerate Logits (MMLU Modified)
echo "[3/4] Running: Regenerate Logits (MMLU)..."
python get_answer_regenerate_logits.py \
    --data "$DATA_DIR" \
    --model "$MODEL_NAME" \
    --model_dir "$MODEL_PATH" \
    --hs "$HS_TYPE" \
    --size "$MODEL_SIZE" \
    --type "$TYPE" \
    --percentage "$PERCENTAGE" \
    --configs $CONFIGS \
    --mask_type "$MASK_TYPE" \
    --ans_file mmlu_mdf \
    --suite "$SUITE"

# 4. Regenerate Logits (MMLU-Pro Modified)
echo "[4/4] Running: Regenerate Logits (MMLU-Pro)..."
python3 get_answer_regenerate_logits_mmlupro.py \
    --data "$DATA_DIR" \
    --model "$MODEL_NAME" \
    --model_dir "$MODEL_PATH" \
    --hs "$HS_TYPE" \
    --size "$MODEL_SIZE" \
    --type "$TYPE" \
    --percentage "$PERCENTAGE" \
    --configs $CONFIGS \
    --mask_type "$MASK_TYPE" \
    --test_file mmlupro/mmlupro_test.json \
    --ans_file answer/answer_mdf_mmlupro \
    --suite "$SUITE"

echo "=================================================="
echo "Finished！"
echo "=================================================="