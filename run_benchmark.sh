#!/bin/bash

#SBATCH -A MST114558                                # Account ID
#SBATCH --job-name=mistral3_benchmark            # Job name
#SBATCH --output=./execution/output_%j.log      # Standard output log
#SBATCH --error=./execution/error_%j.log        # Error output log
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --gres=gpu:1                            # Number of GPUs (14B needs 1 H100)
#SBATCH --cpus-per-task=4                       # Number of CPUs
#SBATCH --time=4:00:00                         # Maximum runtime
#SBATCH --partition=normal                      # Partition
#SBATCH --mail-type=ALL                         # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com       # Email address

# ==================== Configuration ====================
USERNAME="d12922004"
MODEL_NAME="mistral3"
MODEL_DIR="/work/${USERNAME}/models/Mistral3-14B"
MODEL_SIZE="14B"
TYPE="non"
HS_PREFIX="mistral3"

# Role and template configuration
ROLES="neutral"
SUITE="default"
# No E option (A-D only)

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Alpha and layer range configurations (format: alpha-start-end)
# TODO: Update CONFIGS for mistral3-14B
CONFIGS="4-8-24 4-15-24 neg4-8-24 neg4-15-24"
# ==================== Benchmarks ====================
# MMLU-Pro style benchmarks
declare -A MMLUPRO_BENCHMARKS=(
    ["mmlupro"]="benchmark/mmlupro_test"
    ["factor"]="benchmark/factor_mc"
    ["gpqa"]="benchmark/gpqa_train"
    ["arlsat"]="benchmark/arlsat_all"
    ["logiqa"]="benchmark/logiqa_mrc"
)

# TruthfulQA benchmarks
declare -A TQA_FILES=(
    ["mc1"]="benchmark/truthfulqa_mc1_validation_shuf.json"
    ["mc2"]="benchmark/truthfulqa_mc2_validation_shuf.json"
)

# ==================== Paths ====================
WORK_DIR="/work/${USERNAME}/RolePlaying"
BASE_DIR="${WORK_DIR}/components"
CONDA_ENV="roleplaying"

# ==================== Environment ====================
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Model dir: ${MODEL_DIR}"
echo "Roles: ${ROLES}"
echo "Configs: ${CONFIGS}"
echo "=================================================="

# Load miniconda
ml load miniconda3

# Activate conda environment
conda activate ${CONDA_ENV}

# Check environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi

cd ${WORK_DIR}

# ==================== 1. MMLU (standard) ====================
echo ""
echo "=========================================="
echo "[1/6] Running MMLU (original)"
echo "=========================================="

python get_answer_logits.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --roles "${ROLES}" \
    --ans_file "answer_mmlu" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}"

echo "[Done] MMLU original"

# ==================== 2. MMLU Regenerate ====================
echo ""
echo "=========================================="
echo "[2/6] Running MMLU (regenerate)"
echo "=========================================="

python get_answer_regenerate_logits.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --ans_file "answer_mdf_mmlu" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}"

echo "[Done] MMLU regenerate"

# ==================== 3. MMLU-Pro style benchmarks (original) ====================
echo ""
echo "=========================================="
echo "[3/6] Running MMLU-Pro style benchmarks (original)"
echo "=========================================="

for NAME in "${!MMLUPRO_BENCHMARKS[@]}"; do
    TEST_FILE="${MMLUPRO_BENCHMARKS[$NAME]}"
    echo ""
    echo "[Running] ${NAME} (original)"
    echo "Test file: ${TEST_FILE}.json"

    python get_answer_logits_mmlupro.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --test_file "${TEST_FILE}.json" \
        --ans_file "answer_${NAME}" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}"

    echo "[Done] ${NAME} original"
done

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

# ==================== 5. TruthfulQA (original) ====================
echo ""
echo "=========================================="
echo "[5/6] Running TruthfulQA (original)"
echo "=========================================="

for MODE in "${!TQA_FILES[@]}"; do
    TEST_FILE="${TQA_FILES[$MODE]}"
    echo ""
    echo "[Running] TruthfulQA ${MODE} (original)"
    echo "Test file: ${TEST_FILE}"

    python get_answer_logits_tqa.py \
        --mode "${MODE}" \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --ans_file "answer_tqa" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --test_file "${TEST_FILE}"

    echo "[Done] TruthfulQA ${MODE} original"
done

# ==================== 6. TruthfulQA (regenerate) ====================
echo ""
echo "=========================================="
echo "[6/6] Running TruthfulQA (regenerate)"
echo "=========================================="

for MODE in "${!TQA_FILES[@]}"; do
    TEST_FILE="${TQA_FILES[$MODE]}"
    echo ""
    echo "[Running] TruthfulQA ${MODE} (regenerate)"
    echo "Test file: ${TEST_FILE}"

    python get_answer_regenerate_logits_tqa.py \
        --mode "${MODE}" \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --percentage "${PERCENTAGE}" \
        --configs ${CONFIGS} \
        --mask_type "${MASK_TYPE}" \
        --ans_file "answer_mdf_tqa" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --test_file "${TEST_FILE}"

    echo "[Done] TruthfulQA ${MODE} regenerate"
done

echo ""
echo "=================================================="
echo "All benchmarks finished at: $(date)"
echo "=================================================="