#!/bin/bash

# ==================== Supplement Missing Roles (qwen3) ====================
# Supplements the following missing roles in mdf_4 and mdf_-4:
#
#   All tasks (mdf_4 + mdf_-4): confident, unconfident, student, person
#       - mmlupro: {task} student, person, confident, unconfident
#       - others:  student, person, confident, unconfident
#
# Usage: bash run_benchmark_roles_qwen3.sh

# ==================== Model Configuration ====================
MODEL_NAME="qwen3"
MODEL_DIR="Qwen/Qwen3-8B"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="qwen3"
SUITE="default"
DATA="data1"
MASK_TYPE="nmd"
PERCENTAGE=0.5

# qwen3-8B: alpha=4, layers 17-26
CONFIGS_POS4="4-17-26"
CONFIGS_NEG4="neg4-17-26"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Benchmarks ====================
declare -A MMLUPRO_BENCHMARKS=(
    ["mmlupro"]="benchmark/mmlupro_test"
    ["factor"]="benchmark/factor_mc"
    ["gpqa"]="benchmark/gpqa_train"
    ["arlsat"]="benchmark/arlsat_all"
    ["logiqa"]="benchmark/logiqa_mrc"
)
declare -A TQA_FILES=(
    ["mc1"]="benchmark/truthfulqa_mc1_validation_shuf.json"
    ["mc2"]="benchmark/truthfulqa_mc2_validation_shuf.json"
)

# ==================== Environment ====================
echo "=================================================="
echo "Supplement Missing Roles — qwen3"
echo "Start time: $(date)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "=================================================="

cd "${WORK_DIR}"

# ==================== 1. MMLU-Pro style: confident, unconfident, student, person (+4 and -4) ====================
echo ""
echo "=========================================="
echo "[1] MMLU-Pro style: supplement confident, unconfident, student, person (+4 and -4)"
echo "=========================================="

for TASK_NAME in "${!MMLUPRO_BENCHMARKS[@]}"; do
    TEST_FILE="${MMLUPRO_BENCHMARKS[$TASK_NAME]}"

    if [ "${TASK_NAME}" == "mmlupro" ]; then
        ROLES_SUPPLEMENT="confident,unconfident,{task} student,person"
    else
        ROLES_SUPPLEMENT="confident,unconfident,student,person"
    fi

    echo ""
    echo "---------- ${TASK_NAME}: confident + unconfident + student + person (+4 and -4) ----------"

    python get_answer_regenerate_logits_mmlupro.py \
        --data "${DATA}" \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --percentage "${PERCENTAGE}" \
        --configs ${CONFIGS_POS4} ${CONFIGS_NEG4} \
        --mask_type "${MASK_TYPE}" \
        --test_file "${TEST_FILE}.json" \
        --ans_file "answer_mdf_${TASK_NAME}" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES_SUPPLEMENT}"

    if [ $? -eq 0 ]; then
        echo "[✓ Done] ${TASK_NAME} confident + unconfident + student + person"
    else
        echo "[✗ Failed] ${TASK_NAME} confident + unconfident + student + person"
    fi
done

# ==================== 2. TruthfulQA: confident, unconfident, student, person (+4 and -4) ====================
echo ""
echo "=========================================="
echo "[2] TruthfulQA: supplement confident, unconfident, student, person (+4 and -4)"
echo "=========================================="

for MODE in "${!TQA_FILES[@]}"; do
    TEST_FILE="${TQA_FILES[$MODE]}"
    echo ""
    echo "---------- TruthfulQA ${MODE}: confident + unconfident + student + person (+4 and -4) ----------"

    python get_answer_regenerate_logits_tqa.py \
        --data "${DATA}" \
        --mode "${MODE}" \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --percentage "${PERCENTAGE}" \
        --configs ${CONFIGS_POS4} ${CONFIGS_NEG4} \
        --mask_type "${MASK_TYPE}" \
        --ans_file "answer_mdf_tqa" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "confident,unconfident,student,person" \
        --test_file "${TEST_FILE}"

    if [ $? -eq 0 ]; then
        echo "[✓ Done] TruthfulQA ${MODE} confident + unconfident + student + person"
    else
        echo "[✗ Failed] TruthfulQA ${MODE} confident + unconfident + student + person"
    fi
done

# ==================== Summary ====================
echo ""
echo "=================================================="
echo "Supplement finished at: $(date)"
echo "=================================================="
