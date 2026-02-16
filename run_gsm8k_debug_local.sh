#!/bin/bash
# ==================== GSM8K Debug Script (Local) ====================
# This script tests GSM8K generation with only the first 2 samples
# For local development/debugging
#
# Usage: bash run_gsm8k_debug_local.sh

# ==================== Configuration ====================
MODEL="llama3"
SIZE="8B"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
DATA="data1"

# Role and template configuration
ROLES="neutral"
SUITE="default"

# Generation parameters
MAX_NEW_TOKENS=96
TEMPERATURE=0.0

# GSM8K data file
GSM8K_FILE="benchmark/gsm8k_test.json"

# ==================== Paths ====================
WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Environment ====================
echo "=================================================="
echo "GSM8K Debug Script (Local)"
echo "Start time: $(date)"
echo "=================================================="
echo "Model: ${MODEL} (${SIZE})"
echo "Model path: ${MODEL_PATH}"
echo "Roles: ${ROLES}"
echo "Samples: First 2 only (debug mode)"
echo "=================================================="

cd ${WORK_DIR}

# ==================== 1. GSM8K (baseline) - Debug ====================
echo ""
echo "=========================================="
echo "Running GSM8K (original) - First 2 samples"
echo "=========================================="

python get_answer_gsm8k.py \
    --model "${MODEL}" \
    --model_dir "${MODEL_PATH}" \
    --size "${SIZE}" \
    --test_file "benchmark/gsm8k_debug.json" \
    --ans_file "answer_gsm8k_debug" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE}

echo "[Done] GSM8K debug"

echo ""
echo "=================================================="
echo "Debug run finished at: $(date)"
echo "=================================================="
echo ""
echo "Check the output in: ${BASE_DIR}/llama3/answer_gsm8k_debug/"
