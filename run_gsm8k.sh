#!/bin/bash
# ==================== GSM8K Local Run Script ====================
# For running on lab server (not NCHC)
# Model: Mistral3-8B
#
# Runs GSM8K benchmarks:
#   1. GSM8K (original/baseline)
#   2. GSM8K (regenerate)
#
# Usage: bash run_gsm8k_local.sh

# ==================== Configuration ====================
MODEL_NAME="mistral"
MODEL_DIR="mistralai/Mistral-7B-Instruct-v0.3"  # Direct HuggingFace download
MODEL_SIZE="7B"
TYPE="non"
HS_PREFIX="mistral"
DATA="data1"

# Role and template configuration
ROLES="neutral"
SUITE="default"

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Generation parameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.0

# Alpha and layer range configurations (format: alpha-start-end)
CONFIGS="4-14-22 neg4-14-22"

# GSM8K data file
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# ==================== Paths ====================
WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Environment ====================
echo "=================================================="
echo "Start time: $(date)"
echo "=================================================="
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Model dir: ${MODEL_DIR}"
echo "Roles: ${ROLES}"
echo "Configs: ${CONFIGS}"
echo "Task: GSM8K"
echo "=================================================="

cd ${WORK_DIR}

# # ==================== 1. GSM8K (original/baseline) ====================
# echo ""
# echo "=========================================="
# echo "[1/2] Running GSM8K (original)"
# echo "=========================================="

# python get_answer_gsm8k.py \
#     --model "${MODEL_NAME}" \
#     --model_dir "${MODEL_DIR}" \
#     --size "${MODEL_SIZE}" \
#     --test_file "${GSM8K_FILE}" \
#     --ans_file "answer_gsm8k" \
#     --suite "${SUITE}" \
#     --base_dir "${BASE_DIR}" \
#     --roles "${ROLES}" \
#     --max_new_tokens ${MAX_NEW_TOKENS} \
#     --temperature ${TEMPERATURE}

# echo "[Done] GSM8K original"

# ==================== 2. GSM8K (regenerate) ====================
echo ""
echo "=========================================="
echo "[2/2] Running GSM8K (regenerate)"
echo "=========================================="

python get_answer_regenerate_gsm8k.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --test_file "${GSM8K_FILE}" \
    --ans_file "answer_mdf_gsm8k" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE}

echo "[Done] GSM8K regenerate"

echo ""
echo "=================================================="
echo "All GSM8K benchmarks finished at: $(date)"
echo "=================================================="
