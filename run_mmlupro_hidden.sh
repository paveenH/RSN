#!/bin/bash

# ==================== MMLU-Pro Hidden States Extraction ====================
# Extract hidden states and answers for both Llama3 and Qwen3
#
# Usage: bash run_mmlupro_hidden.sh

# ==================== Configuration ====================

# Models to run (comment out to skip)
RUN_LLAMA3=true
RUN_QWEN3=true

# ==================== Llama3-8B Configuration ====================
LLAMA3_MODEL_NAME="llama3"
LLAMA3_MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
LLAMA3_MODEL_SIZE="8B"
LLAMA3_TYPE="non"
LLAMA3_ROLES="neutral"
LLAMA3_SUITE="default"
LLAMA3_DATA="data1"

# ==================== Qwen3-8B Configuration ====================
QWEN3_MODEL_NAME="qwen3"
QWEN3_MODEL_DIR="Qwen/Qwen3-8B"
QWEN3_MODEL_SIZE="8B"
QWEN3_TYPE="neutral"
QWEN3_ROLES="neutral"
QWEN3_SUITE="default"
QWEN3_DATA="data1"

# ==================== Common Configuration ====================
TEST_FILE="benchmark/mmlupro_test.json"
ANS_FILE="answers"
BASE_DIR=""  # Leave empty to use /{DATA}/paveen/RolePlaying/components
USE_E=""     # Set to "--use_E" to enable 5-choice template
COT=""       # Set to "--cot" to enable chain-of-thought

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"

# ==================== Helper Functions ====================
run_model() {
    local MODEL_NAME=$1
    local MODEL_DIR=$2
    local MODEL_SIZE=$3
    local TYPE=$4
    local ROLES=$5
    local SUITE=$6
    local DATA=$7

    echo ""
    echo "=========================================="
    echo "Running: ${MODEL_NAME} (${MODEL_SIZE})"
    echo "=========================================="
    echo "Model dir: ${MODEL_DIR}"
    echo "Roles: ${ROLES}"
    echo "Test file: ${TEST_FILE}"
    echo ""

    python get_answer_logits_mmlupro.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --roles "${ROLES}" \
        --test_file "${TEST_FILE}" \
        --ans_file "${ANS_FILE}" \
        --suite "${SUITE}" \
        --data "${DATA}" \
        --save \
        ${USE_E} \
        ${COT} \
        ${BASE_DIR:+--base_dir "$BASE_DIR"}

    if [ $? -eq 0 ]; then
        echo ""
        echo "[✓ Done] ${MODEL_NAME}"
    else
        echo ""
        echo "[✗ Failed] ${MODEL_NAME}"
        return 1
    fi
}

# ==================== Environment ====================
echo "=================================================="
echo "MMLU-Pro Hidden States Extraction"
echo "Start time: $(date)"
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

# ==================== Run Models ====================

if [ "${RUN_LLAMA3}" = "true" ]; then
    echo ""
    echo "========== LLAMA3 =========="
    run_model \
        "${LLAMA3_MODEL_NAME}" \
        "${LLAMA3_MODEL_DIR}" \
        "${LLAMA3_MODEL_SIZE}" \
        "${LLAMA3_TYPE}" \
        "${LLAMA3_ROLES}" \
        "${LLAMA3_SUITE}" \
        "${LLAMA3_DATA}"

    if [ $? -ne 0 ]; then
        echo "Llama3 extraction failed!"
    fi
fi

if [ "${RUN_QWEN3}" = "true" ]; then
    echo ""
    echo "========== QWEN3 =========="
    run_model \
        "${QWEN3_MODEL_NAME}" \
        "${QWEN3_MODEL_DIR}" \
        "${QWEN3_MODEL_SIZE}" \
        "${QWEN3_TYPE}" \
        "${QWEN3_ROLES}" \
        "${QWEN3_SUITE}" \
        "${QWEN3_DATA}"

    if [ $? -ne 0 ]; then
        echo "Qwen3 extraction failed!"
    fi
fi

# ==================== Summary ====================
echo ""
echo "=================================================="
echo "Extraction finished at: $(date)"
echo "=================================================="
