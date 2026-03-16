#!/bin/bash

# ==================== All Benchmarks Hidden States Extraction (Llama3) ====================
# Extract hidden states and answers for Llama3-8B
# Covers: MMLU-Pro style benchmarks + TruthfulQA
#
# Usage: bash run_all_hidden_llama3.sh

# ==================== Model Configuration ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
ROLES="neutral"
SUITE="default"
DATA="data1"

# ==================== Common Configuration ====================
ANS_FILE="answers"
BASE_DIR=""  # Leave empty to use /{DATA}/paveen/RolePlaying/components
USE_E=""     # Set to "--use_E" to enable 5-choice template
COT=""       # Set to "--cot" to enable chain-of-thought

# ==================== Benchmarks ====================
# MMLU-Pro style benchmarks (task_name → test_file)
declare -A MMLUPRO_BENCHMARKS=(
    ["mmlupro"]="benchmark/mmlupro_test"
    ["factor"]="benchmark/factor_mc"
    ["gpqa"]="benchmark/gpqa_train"
    ["arlsat"]="benchmark/arlsat_all"
    ["logiqa"]="benchmark/logiqa_mrc"
)

# TruthfulQA modes
declare -a TQA_MODES=("mc1" "mc2")

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"

# ==================== Environment ====================
echo "=================================================="
echo "All Benchmarks Hidden States Extraction (Llama3)"
echo "Start time: $(date)"
echo "=================================================="
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Model dir: ${MODEL_DIR}"
echo "Roles: ${ROLES}"
echo "=================================================="
echo ""

# Change to work directory
if [ -d "${WORK_DIR}" ]; then
    cd "${WORK_DIR}"
    echo "Working directory: $(pwd)"
else
    echo "Warning: WORK_DIR not found: ${WORK_DIR}"
    echo "Using current directory instead"
fi
echo ""

# ==================== 1. MMLU-Pro Style Benchmarks ====================
echo "=========================================="
echo "[1] Running MMLU-Pro style benchmarks"
echo "=========================================="

for TASK_NAME in "${!MMLUPRO_BENCHMARKS[@]}"; do
    TEST_FILE="${MMLUPRO_BENCHMARKS[$TASK_NAME]}"
    echo ""
    echo "---------- ${TASK_NAME} ----------"
    echo "Test file: ${TEST_FILE}.json"

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
        --save \
        ${USE_E} \
        ${COT} \
        ${BASE_DIR:+--base_dir "$BASE_DIR"}

    if [ $? -eq 0 ]; then
        echo "[✓ Done] ${TASK_NAME}"
    else
        echo "[✗ Failed] ${TASK_NAME}"
    fi
done

# ==================== 2. TruthfulQA ====================
echo ""
echo "=========================================="
echo "[2] Running TruthfulQA"
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
        --roles "${ROLES}" \
        --ans_file "answer_tqa" \
        --suite "${SUITE}" \
        --data "${DATA}" \
        --save \
        ${USE_E} \
        ${COT} \
        ${BASE_DIR:+--base_dir "$BASE_DIR"}

    if [ $? -eq 0 ]; then
        echo "[✓ Done] TruthfulQA ${MODE}"
    else
        echo "[✗ Failed] TruthfulQA ${MODE}"
    fi
done

# ==================== Summary ====================
echo ""
echo "=================================================="
echo "All extractions finished at: $(date)"
echo "=================================================="
