#!/bin/bash
# ==================== Benchmark Local Run Script ====================
# For running on lab server (not NCHC)
# Model: Mistral3-8B-IT (mistralai/Mistral-Small-3.1-24B-Instruct-2503 → 8B)
#
# Runs all 6 benchmarks:
#   1. MMLU (original)
#   2. MMLU (regenerate)
#   3. MMLU-Pro style benchmarks (original)
#   4. MMLU-Pro style benchmarks (regenerate)
#   5. TruthfulQA (original)
#   6. TruthfulQA (regenerate)
#
# Usage: bash run_benchmark_local.sh

# ==================== Configuration ====================
MODEL_NAME="mistral3"
MODEL_DIR="mistralai/Ministral-3-8B-Reasoning-2512"  # Direct HuggingFace download
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="mistral3"
DATA="data1"

# Role and template configuration
ROLES="neutral"
SUITE="default"
# No E option (A-D only)

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Alpha and layer range configurations (format: alpha-start-end)
# TODO: Update CONFIGS for mistral3-8B
CONFIGS="4-11-22 neg4-11-22"

# ==================== Benchmarks ====================
# MMLU-Pro style benchmarks
declare -A MMLUPRO_BENCHMARKS=(
    # ["mmlupro"]="benchmark/mmlupro_test"
    ["factor"]="benchmark/factor_mc"
    # ["gpqa"]="benchmark/gpqa_train"
    # ["arlsat"]="benchmark/arlsat_all"
    # ["logiqa"]="benchmark/logiqa_mrc"
)

# TruthfulQA benchmarks
declare -A TQA_FILES=(
    ["mc1"]="benchmark/truthfulqa_mc1_validation_shuf.json"
    ["mc2"]="benchmark/truthfulqa_mc2_validation_shuf.json"
)

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
echo "=================================================="

cd ${WORK_DIR}

# # ==================== 1. MMLU (standard) ====================
# echo ""
# echo "=========================================="
# echo "[1/6] Running MMLU (original)"
# echo "=========================================="

# python get_answer_logits.py \
#     --data "${DATA}" \
#     --model "${MODEL_NAME}" \
#     --model_dir "${MODEL_DIR}" \
#     --size "${MODEL_SIZE}" \
#     --type "${TYPE}" \
#     --roles "${ROLES}" \
#     --ans_file "answer_mmlu" \
#     --suite "${SUITE}" \
#     --base_dir "${BASE_DIR}"

# echo "[Done] MMLU original"

# # ==================== 2. MMLU Regenerate ====================
# echo ""
# echo "=========================================="
# echo "[2/6] Running MMLU (regenerate)"
# echo "=========================================="

# python get_answer_regenerate_logits.py \
#     --data "${DATA}" \
#     --model "${MODEL_NAME}" \
#     --model_dir "${MODEL_DIR}" \
#     --hs "${HS_PREFIX}" \
#     --size "${MODEL_SIZE}" \
#     --type "${TYPE}" \
#     --percentage "${PERCENTAGE}" \
#     --configs ${CONFIGS} \
#     --mask_type "${MASK_TYPE}" \
#     --ans_file "answer_mdf_mmlu" \
#     --suite "${SUITE}" \
#     --base_dir "${BASE_DIR}" \
#     --roles "${ROLES}"

# echo "[Done] MMLU regenerate"

# # ==================== 3. MMLU-Pro style benchmarks (original) ====================
# echo ""
# echo "=========================================="
# echo "[3/6] Running MMLU-Pro style benchmarks (original)"
# echo "=========================================="

# for NAME in "${!MMLUPRO_BENCHMARKS[@]}"; do
#     TEST_FILE="${MMLUPRO_BENCHMARKS[$NAME]}"
#     echo ""
#     echo "[Running] ${NAME} (original)"
#     echo "Test file: ${TEST_FILE}.json"

#     python get_answer_logits_mmlupro.py \
#         --data "${DATA}" \
#         --model "${MODEL_NAME}" \
#         --model_dir "${MODEL_DIR}" \
#         --size "${MODEL_SIZE}" \
#         --type "${TYPE}" \
#         --test_file "${TEST_FILE}.json" \
#         --ans_file "answer_${NAME}" \
#         --suite "${SUITE}" \
#         --base_dir "${BASE_DIR}" \
#         --roles "${ROLES}"

#     echo "[Done] ${NAME} original"
# done

# ==================== 4. MMLU-Pro style benchmarks (regenerate) ====================
echo ""
echo "=========================================="
echo "[4/6] Running MMLU-Pro style benchmarks (regenerate)"
echo "=========================================="

for NAME in "${!MMLUPRO_BENCHMARKS[@]}"; do
    TEST_FILE="${MMLUPRO_BENCHMARKS[$NAME]}"
    echo ""
    echo "[Running] ${NAME} (regenerate)"
    echo "Test file: ${TEST_FILE}.json"

    python get_answer_regenerate_logits_mmlupro.py \
        --data "${DATA}" \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --percentage "${PERCENTAGE}" \
        --configs ${CONFIGS} \
        --mask_type "${MASK_TYPE}" \
        --test_file "${TEST_FILE}.json" \
        --ans_file "answer_mdf_${NAME}" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}"

    echo "[Done] ${NAME} regenerate"
done

# # ==================== 5. TruthfulQA (original) ====================
# echo ""
# echo "=========================================="
# echo "[5/6] Running TruthfulQA (original)"
# echo "=========================================="

# for MODE in "${!TQA_FILES[@]}"; do
#     TEST_FILE="${TQA_FILES[$MODE]}"
#     echo ""
#     echo "[Running] TruthfulQA ${MODE} (original)"
#     echo "Test file: ${TEST_FILE}"

#     python get_answer_logits_tqa.py \
#         --data "${DATA}" \
#         --mode "${MODE}" \
#         --model "${MODEL_NAME}" \
#         --model_dir "${MODEL_DIR}" \
#         --size "${MODEL_SIZE}" \
#         --ans_file "answer_tqa" \
#         --suite "${SUITE}" \
#         --base_dir "${BASE_DIR}" \
#         --roles "${ROLES}" \
#         --test_file "${TEST_FILE}"

#     echo "[Done] TruthfulQA ${MODE} original"
# done

# # ==================== 6. TruthfulQA (regenerate) ====================
# echo ""
# echo "=========================================="
# echo "[6/6] Running TruthfulQA (regenerate)"
# echo "=========================================="

# for MODE in "${!TQA_FILES[@]}"; do
#     TEST_FILE="${TQA_FILES[$MODE]}"
#     echo ""
#     echo "[Running] TruthfulQA ${MODE} (regenerate)"
#     echo "Test file: ${TEST_FILE}"

#     python get_answer_regenerate_logits_tqa.py \
#         --data "${DATA}" \
#         --mode "${MODE}" \
#         --model "${MODEL_NAME}" \
#         --model_dir "${MODEL_DIR}" \
#         --hs "${HS_PREFIX}" \
#         --size "${MODEL_SIZE}" \
#         --type "${TYPE}" \
#         --percentage "${PERCENTAGE}" \
#         --configs ${CONFIGS} \
#         --mask_type "${MASK_TYPE}" \
#         --ans_file "answer_mdf_tqa" \
#         --suite "${SUITE}" \
#         --base_dir "${BASE_DIR}" \
#         --roles "${ROLES}" \
#         --test_file "${TEST_FILE}"

#     echo "[Done] TruthfulQA ${MODE} regenerate"
# done

# echo ""
# echo "=================================================="
# echo "All benchmarks finished at: $(date)"
# echo "=================================================="
