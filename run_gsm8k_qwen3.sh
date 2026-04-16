#!/bin/bash
# ==================== GSM8K Qwen3 Re-run Script ====================
# Re-runs Qwen3-8B GSM8K with temperature=0.7, top_p=0.8 (official
# non-thinking mode recommendation) instead of greedy decoding.
#
# Runs both conditions:
#   1. Without CoT (no --cot flag)  → answer_gsm8k / answer_mdf_gsm8k
#   2. With CoT    (--cot flag)     → answer_gsm8k_cot / answer_mdf_gsm8k_cot
#
# Three conditions each: orig / mdf_4 / mdf_-4
# Configs: 4-17-26 / neg4-17-26
#
# Usage: bash run_gsm8k_qwen3.sh

# ==================== Config ====================
MODEL_NAME="qwen3"
MODEL_DIR="Qwen/Qwen3-8B"
MODEL_SIZE="8B"
HS_PREFIX="qwen3"
TYPE="non"
PERCENTAGE=0.5
MASK_TYPE="nmd"
CONFIGS="4-17-26 neg4-17-26"
ROLES="neutral"
SUITE="default"
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# Qwen3 official non-thinking mode parameters
TEMPERATURE=0.7
TOP_P=0.8
MAX_NEW_TOKENS=512
BATCH_SIZE=24

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Start ====================
echo "=================================================="
echo "GSM8K Qwen3 Re-run (temperature=0.7, top_p=0.8)"
echo "Start time: $(date)"
echo "=================================================="

cd "${WORK_DIR}"

# ==================== [1] Without CoT — Baseline ====================
echo ""
echo "=========================================="
echo "[1/4] GSM8K baseline (no CoT) — ${MODEL_NAME}"
echo "=========================================="

python get_answer_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --size       "${MODEL_SIZE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_gsm8k" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --top_p          ${TOP_P} \
    --batch_size     ${BATCH_SIZE}

if [ $? -eq 0 ]; then
    echo "[✓ Done] baseline no-CoT — ${MODEL_NAME}"
else
    echo "[✗ Failed] baseline no-CoT — ${MODEL_NAME}"
    exit 1
fi

# ==================== [2] Without CoT — Regenerate (+4 / -4) ====================
echo ""
echo "=========================================="
echo "[2/4] GSM8K regenerate (no CoT) — ${MODEL_NAME}"
echo "=========================================="

python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    ${CONFIGS} \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_mdf_gsm8k" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --top_p          ${TOP_P} \
    --batch_size     ${BATCH_SIZE}

if [ $? -eq 0 ]; then
    echo "[✓ Done] regenerate no-CoT — ${MODEL_NAME}"
else
    echo "[✗ Failed] regenerate no-CoT — ${MODEL_NAME}"
    exit 1
fi

# ==================== [3] With CoT — Baseline ====================
echo ""
echo "=========================================="
echo "[3/4] GSM8K baseline (CoT) — ${MODEL_NAME}"
echo "=========================================="

python get_answer_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --size       "${MODEL_SIZE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_gsm8k_cot" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --top_p          ${TOP_P} \
    --batch_size     ${BATCH_SIZE} \
    --cot

if [ $? -eq 0 ]; then
    echo "[✓ Done] baseline CoT — ${MODEL_NAME}"
else
    echo "[✗ Failed] baseline CoT — ${MODEL_NAME}"
    exit 1
fi

# ==================== [4] With CoT — Regenerate (+4 / -4) ====================
echo ""
echo "=========================================="
echo "[4/4] GSM8K regenerate (CoT) — ${MODEL_NAME}"
echo "=========================================="

python get_answer_regenerate_gsm8k.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS_PREFIX}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs    ${CONFIGS} \
    --mask_type  "${MASK_TYPE}" \
    --test_file  "${GSM8K_FILE}" \
    --ans_file   "answer_mdf_gsm8k_cot" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature    ${TEMPERATURE} \
    --top_p          ${TOP_P} \
    --batch_size     ${BATCH_SIZE} \
    --cot

if [ $? -eq 0 ]; then
    echo "[✓ Done] regenerate CoT — ${MODEL_NAME}"
else
    echo "[✗ Failed] regenerate CoT — ${MODEL_NAME}"
    exit 1
fi

echo ""
echo "=================================================="
echo "All Qwen3 GSM8K runs finished at: $(date)"
echo "=================================================="
