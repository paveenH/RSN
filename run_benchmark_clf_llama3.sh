#!/bin/bash
# ==================== Classifier-Guided Steering Benchmark (llama3) ====================
# Three-way comparison: no_steer / always_steer / classifier
# Classifier: MMLU-trained PCA-CNN (with residual) from ConfSteer
#
# Usage: bash run_benchmark_clf_llama3.sh

# ==================== Model Configuration ====================
MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
SUITE="default"
DATA="data1"
MASK_TYPE="nmd"
PERCENTAGE=0.5

# llama3-8B: alpha=4, layers 11-20
CONFIGS="4-11-20"

# Roles
ROLES="neutral"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"
CLF_DIR="/data1/paveen/ConfSteer/models/llama3_pca128_mmlu_cnn_k7"

# ==================== Benchmarks ====================
MMLUPRO_TEST="benchmark/mmlupro_test.json"

# ==================== Environment ====================
echo "=================================================="
echo "Classifier-Guided Steering Benchmark — llama3"
echo "Start time: $(date)"
echo "Model : ${MODEL_NAME} (${MODEL_SIZE})"
echo "Clf   : ${CLF_DIR}"
echo "Config: ${CONFIGS}"
echo "Roles : ${ROLES}"
echo "=================================================="

cd "${WORK_DIR}"

# ==================== MMLU-Pro ====================
echo ""
echo "=========================================="
echo "[1] MMLU-Pro (classifier benchmark)"
echo "=========================================="

python get_answer_classifier_mmlupro_mmlu.py \
    --model      "${MODEL_NAME}" \
    --model_dir  "${MODEL_DIR}" \
    --size       "${MODEL_SIZE}" \
    --type       "${TYPE}" \
    --test_file  "${MMLUPRO_TEST}" \
    --ans_file   "answer_clf_mmlupro_mmlu" \
    --clf_dir    "${CLF_DIR}" \
    --hs         "${HS_PREFIX}" \
    --percentage "${PERCENTAGE}" \
    --configs    ${CONFIGS} \
    --mask_type  "${MASK_TYPE}" \
    --suite      "${SUITE}" \
    --base_dir   "${BASE_DIR}" \
    --roles      "${ROLES}"

if [ $? -eq 0 ]; then
    echo "[✓ Done] MMLU-Pro classifier benchmark"
else
    echo "[✗ Failed] MMLU-Pro classifier benchmark"
    exit 1
fi

echo ""
echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
