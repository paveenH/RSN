#!/bin/bash

# ==================== MMLU Hidden States Extraction ====================
# Extract hidden states and answers for MMLU (standard 4-option) tasks.
# Saves H5 files to ConfSteer/HiddenStates for classifier training.
#
# Roles (7):
#   neutral, {task} expert, non {task} expert,
#   confident, unconfident, student, person
#
# Output:
#   ConfSteer/HiddenStates/{model}/mmlu/  — per-role per-task .h5
#   components/{model}/answer_hs_mmlu/    — answer JSON + summary CSV
#
# Usage: bash run_save_hs_mmlu.sh [llama3|qwen3|all]
#   Default: all

TARGET="${1:-all}"

# ==================== Common Configuration ====================
TYPE="non"
SUITE="default"
DATA="data1"
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"
HS_DIR="/data1/paveen/ConfSteer/HiddenStates"
TASK_NAME="mmlu"

ROLES="neutral,{task} expert,non {task} expert,confident,unconfident,student,person"

# ==================== Helper Function ====================
run_model() {
    local MODEL_NAME="$1"
    local MODEL_DIR="$2"
    local MODEL_SIZE="$3"

    echo ""
    echo "=================================================="
    echo "  Model    : ${MODEL_NAME} (${MODEL_SIZE})"
    echo "  Model dir: ${MODEL_DIR}"
    echo "  HS output: ${HS_DIR}/${MODEL_NAME}/${TASK_NAME}/"
    echo "  Start    : $(date)"
    echo "=================================================="

    python get_answer_logits.py \
        --model      "${MODEL_NAME}" \
        --model_dir  "${MODEL_DIR}" \
        --size       "${MODEL_SIZE}" \
        --type       "${TYPE}" \
        --ans_file   "answer_hs_mmlu" \
        --suite      "${SUITE}" \
        --data       "${DATA}" \
        --base_dir   "${BASE_DIR}" \
        --hs_dir     "${HS_DIR}" \
        --task_name  "${TASK_NAME}" \
        --roles      "${ROLES}" \
        --save

    if [ $? -eq 0 ]; then
        echo "[✓ Done] ${MODEL_NAME} MMLU hidden states saved"
    else
        echo "[✗ Failed] ${MODEL_NAME} MMLU hidden states"
        exit 1
    fi

    echo "[✓ Finished] ${MODEL_NAME} at $(date)"
}

# ==================== Change to work directory ====================
if [ -d "${WORK_DIR}" ]; then
    cd "${WORK_DIR}"
    echo "Working directory: $(pwd)"
else
    echo "Warning: WORK_DIR not found: ${WORK_DIR}, using current directory"
fi

# ==================== Run Models ====================
echo "=================================================="
echo "MMLU Hidden States Extraction"
echo "Target: ${TARGET}"
echo "Start time: $(date)"
echo "=================================================="

if [ "${TARGET}" == "llama3" ] || [ "${TARGET}" == "all" ]; then
    run_model "llama3" "meta-llama/Llama-3.1-8B-Instruct" "8B"
fi

if [ "${TARGET}" == "qwen3" ] || [ "${TARGET}" == "all" ]; then
    run_model "qwen3" "Qwen/Qwen3-8B" "8B"
fi

echo ""
echo "=================================================="
echo "All extractions finished at: $(date)"
echo "=================================================="
