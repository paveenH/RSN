#!/bin/bash

# ==================== All Benchmarks Hidden States Extraction ====================
# Extract hidden states and answers for all models (Llama3 + Qwen3)
# Covers: MMLU-Pro style benchmarks + TruthfulQA
#
# Usage: bash run_all_hidden.sh [llama3|qwen3|all]
#   Default: all models

TARGET="${1:-all}"  # llama3 | qwen3 | all

# ==================== Common Configuration ====================
TYPE="non"
SUITE="default"
DATA="data1"
BASE_DIR=""
HS_DIR="/data1/paveen/ConfSteer/HiddenStates"
USE_E=""     # Set to "--use_E" to enable 5-choice template
COT=""       # Set to "--cot" to enable chain-of-thought
WORK_DIR="/data1/paveen/RolePlaying"

# ==================== Roles ====================
ROLES_MMLUPRO="{task} expert,non {task} expert,{task} student,person"
ROLES_GENERIC="expert,non expert,student,person"

# ==================== Benchmarks ====================
declare -A MMLUPRO_BENCHMARKS=(
    ["mmlupro"]="benchmark/mmlupro_test"
    ["factor"]="benchmark/factor_mc"
    ["gpqa"]="benchmark/gpqa_train"
    ["arlsat"]="benchmark/arlsat_all"
    ["logiqa"]="benchmark/logiqa_mrc"
)
declare -a TQA_MODES=("mc1" "mc2")

# ==================== Helper Function ====================
run_model() {
    local MODEL_NAME="$1"
    local MODEL_DIR="$2"
    local MODEL_SIZE="$3"

    echo ""
    echo "=================================================="
    echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
    echo "Model dir: ${MODEL_DIR}"
    echo "HS output: ${HS_DIR}/${MODEL_NAME}/"
    echo "Start time: $(date)"
    echo "=================================================="

    # --- MMLU-Pro style benchmarks ---
    echo ""
    echo "=========================================="
    echo "[1] MMLU-Pro style benchmarks"
    echo "=========================================="

    for TASK_NAME in "${!MMLUPRO_BENCHMARKS[@]}"; do
        TEST_FILE="${MMLUPRO_BENCHMARKS[$TASK_NAME]}"
        echo ""
        echo "---------- ${TASK_NAME} ----------"
        echo "Test file: ${TEST_FILE}.json"

        if [ "${TASK_NAME}" == "mmlupro" ]; then
            ROLES="${ROLES_MMLUPRO}"
        else
            ROLES="${ROLES_GENERIC}"
        fi

        python get_answer_logits_mmlupro.py \
            --model "${MODEL_NAME}" \
            --model_dir "${MODEL_DIR}" \
            --size "${MODEL_SIZE}" \
            --type "${TYPE}" \
            --task_name "${TASK_NAME}" \
            --roles "${ROLES}" \
            --test_file "${TEST_FILE}.json" \
            --ans_file "answer_${TASK_NAME}" \
            --suite "${SUITE}" \
            --data "${DATA}" \
            --hs_dir "${HS_DIR}" \
            --save \
            ${USE_E} \
            ${COT} \
            ${BASE_DIR:+--base_dir "$BASE_DIR"}

        if [ $? -eq 0 ]; then
            echo "[✓ Done] ${MODEL_NAME} / ${TASK_NAME}"
        else
            echo "[✗ Failed] ${MODEL_NAME} / ${TASK_NAME}"
        fi
    done

    # --- TruthfulQA ---
    echo ""
    echo "=========================================="
    echo "[2] TruthfulQA"
    echo "=========================================="

    for MODE in "${TQA_MODES[@]}"; do
        echo ""
        echo "---------- TruthfulQA ${MODE} ----------"

        python get_answer_logits_tqa.py \
            --model "${MODEL_NAME}" \
            --model_dir "${MODEL_DIR}" \
            --size "${MODEL_SIZE}" \
            --type "${TYPE}" \
            --task_name "truthfulqa" \
            --mode "${MODE}" \
            --roles "${ROLES_GENERIC}" \
            --ans_file "answer_tqa" \
            --test_file "benchmark/truthfulqa_${MODE}_validation_shuf.json" \
            --suite "${SUITE}" \
            --data "${DATA}" \
            --hs_dir "${HS_DIR}" \
            --save \
            ${USE_E} \
            ${COT} \
            ${BASE_DIR:+--base_dir "$BASE_DIR"}

        if [ $? -eq 0 ]; then
            echo "[✓ Done] ${MODEL_NAME} / TruthfulQA ${MODE}"
        else
            echo "[✗ Failed] ${MODEL_NAME} / TruthfulQA ${MODE}"
        fi
    done

    echo ""
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
echo "All Benchmarks Hidden States Extraction"
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
