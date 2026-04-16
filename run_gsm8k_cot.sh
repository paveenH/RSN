#!/bin/bash
# ==================== GSM8K CoT Run Script ====================
# Runs GSM8K with explicit chain-of-thought prompting (--cot flag).
# Three conditions per model: no-steering baseline / +4 steering / -4 steering
# Output dirs use _cot suffix to separate from no-CoT runs.
#
# Models  : llama3-8B, qwen3-8B, mistral-7B
# Samples : 300 (gsm8k_test_sample.json, same as no-CoT run)
# Configs : llama3   → 4-11-20 / neg4-11-20
#           qwen3    → 4-17-26 / neg4-17-26
#           mistral  → 4-14-22 / neg4-14-22
#
# Usage: bash run_gsm8k_cot.sh

# ==================== Shared config ====================
ROLES="neutral"
SUITE="default"
MASK_TYPE="nmd"
PERCENTAGE=0.5
MAX_NEW_TOKENS=512
TEMPERATURE=0.0
BATCH_SIZE=24
TYPE="non"
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# ==================== Model configs ====================
# Format: MODEL_NAME|MODEL_DIR|MODEL_SIZE|HS_PREFIX|CONFIGS
MODELS=(
    # "llama3|meta-llama/Llama-3.1-8B-Instruct|8B|llama3|4-11-20 neg4-11-20"
    # "qwen3|Qwen/Qwen3-8B|8B|qwen3|4-17-26 neg4-17-26"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3|7B|mistral|4-14-22 neg4-14-22"
)

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Environment ====================
echo "=================================================="
echo "GSM8K CoT Benchmark"
echo "Start time: $(date)"
echo "Roles : ${ROLES}"
echo "=================================================="

cd "${WORK_DIR}"

TOTAL=$(( ${#MODELS[@]} * 2 ))
STEP=1

for MODEL_CFG in "${MODELS[@]}"; do
    IFS="|" read -r MODEL_NAME MODEL_DIR MODEL_SIZE HS_PREFIX CONFIGS <<< "${MODEL_CFG}"

    echo ""
    echo "####################################################"
    echo "# Model: ${MODEL_NAME} (${MODEL_SIZE})"
    echo "# Configs: ${CONFIGS}"
    echo "####################################################"

    # ==================== [1] Baseline (no steering, CoT) ====================
    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] GSM8K baseline CoT — ${MODEL_NAME}"
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
        --batch_size     ${BATCH_SIZE} \
        --cot

    if [ $? -eq 0 ]; then
        echo "[✓ Done] baseline CoT — ${MODEL_NAME}"
    else
        echo "[✗ Failed] baseline CoT — ${MODEL_NAME}"
        exit 1
    fi
    STEP=$((STEP + 1))

    # ==================== [2] Regenerate (+4 / -4, CoT) ====================
    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] GSM8K regenerate CoT — ${MODEL_NAME}"
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
        --batch_size     ${BATCH_SIZE} \
        --cot

    if [ $? -eq 0 ]; then
        echo "[✓ Done] regenerate CoT — ${MODEL_NAME}"
    else
        echo "[✗ Failed] regenerate CoT — ${MODEL_NAME}"
        exit 1
    fi
    STEP=$((STEP + 1))

done

echo ""
echo "=================================================="
echo "All GSM8K CoT runs finished at: $(date)"
echo "=================================================="
