#!/bin/bash
# ==================== GSM8K Alpha Scan — Llama3-8B ====================
# Scans alpha values ±2 and ±8 (±4 and 0 already exist).
# Runs both with-CoT and without-CoT conditions.
#
# Existing results (skip):  0-11-20 / 4-11-20 / neg4-11-20
# New configs:              2-11-20 / neg2-11-20 / 8-11-20 / neg8-11-20
#
# Output:
#   without CoT → answer_mdf_gsm8k/      (same as existing non-CoT runs)
#   with CoT    → answer_mdf_gsm8k_cot/  (same as existing CoT runs)
#
# Usage: bash run_gsm8k_alpha_scan.sh

# ==================== Shared config ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"
MASK_TYPE="nmd"
PERCENTAGE=0.5
ROLES="neutral"
SUITE="default"
MAX_NEW_TOKENS=512
TEMPERATURE=0.0
BATCH_SIZE=24
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# New alpha configs only (0 / ±4 already done)
CONFIGS="2-11-20 neg2-11-20 8-11-20 neg8-11-20"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Run ====================
echo "=================================================="
echo "GSM8K Alpha Scan — Llama3-8B"
echo "Configs: ${CONFIGS}"
echo "Start time: $(date)"
echo "=================================================="

cd "${WORK_DIR}"

# ──────────────────────────────────────────────────────
# 1. Without CoT
# ──────────────────────────────────────────────────────
echo ""
echo "####################################################"
echo "# [1/2] Without CoT"
echo "####################################################"

python get_answer_regenerate_gsm8k.py \
    --model          "${MODEL_NAME}" \
    --model_dir      "${MODEL_DIR}" \
    --hs             "${HS_PREFIX}" \
    --size           "${MODEL_SIZE}" \
    --type           "${TYPE}" \
    --percentage     "${PERCENTAGE}" \
    --configs        ${CONFIGS} \
    --mask_type      "${MASK_TYPE}" \
    --test_file      "${GSM8K_FILE}" \
    --ans_file       "answer_mdf_gsm8k" \
    --suite          "${SUITE}" \
    --base_dir       "${BASE_DIR}" \
    --roles          "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE}

if [ $? -eq 0 ]; then
    echo "[✓ Done] Without CoT — alpha scan finished at: $(date)"
else
    echo "[✗ Failed] Without CoT — alpha scan"
    exit 1
fi

# ──────────────────────────────────────────────────────
# 2. With CoT
# ──────────────────────────────────────────────────────
echo ""
echo "####################################################"
echo "# [2/2] With CoT"
echo "####################################################"

python get_answer_regenerate_gsm8k.py \
    --model          "${MODEL_NAME}" \
    --model_dir      "${MODEL_DIR}" \
    --hs             "${HS_PREFIX}" \
    --size           "${MODEL_SIZE}" \
    --type           "${TYPE}" \
    --percentage     "${PERCENTAGE}" \
    --configs        ${CONFIGS} \
    --mask_type      "${MASK_TYPE}" \
    --test_file      "${GSM8K_FILE}" \
    --ans_file       "answer_mdf_gsm8k_cot" \
    --suite          "${SUITE}" \
    --base_dir       "${BASE_DIR}" \
    --roles          "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --batch_size     ${BATCH_SIZE} \
    --cot

if [ $? -eq 0 ]; then
    echo "[✓ Done] With CoT — alpha scan finished at: $(date)"
else
    echo "[✗ Failed] With CoT — alpha scan"
    exit 1
fi

echo ""
echo "=================================================="
echo "All alpha scan runs finished at: $(date)"
echo "=================================================="
