#!/bin/bash

#SBATCH -A MST114558                                # Account ID
#SBATCH --job-name=gsm8k_bench                      # Job name
#SBATCH --output=./execution/output_%j.log      # Standard output log
#SBATCH --error=./execution/error_%j.log        # Error output log
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --gres=gpu:1                            # Number of GPUs
#SBATCH --cpus-per-task=4                       # Number of CPUs
#SBATCH --time=12:00:00                         # Maximum runtime
#SBATCH --partition=normal                      # Partition
#SBATCH --mail-type=ALL                         # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com       # Email address

# ==================== Configuration ====================
USERNAME="d12922004"

# Role and template configuration
ROLES="neutral"
SUITE="default"

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Generation parameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.0

# GSM8K data file
GSM8K_FILE="benchmark/gsm8k_test_sample.json"

# ==================== Model configurations ====================
# Format: MODEL_NAME|MODEL_DIR|MODEL_SIZE|HS_PREFIX|CONFIGS
MODELS=(
    "llama3|/work/${USERNAME}/models/Llama3-8B|8B|llama3|4-11-20 neg4-11-20"
    "qwen3|/work/${USERNAME}/models/Qwen3-8B|8B|qwen3|4-17-26 neg4-17-26"
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
echo "Models: llama3-8B, qwen3-8B"
echo "Task: GSM8K"
echo "Roles: ${ROLES}"
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

# ==================== Run each model ====================
STEP=1
TOTAL=$(( ${#MODELS[@]} * 2 ))

for MODEL_CFG in "${MODELS[@]}"; do
    IFS="|" read -r MODEL_NAME MODEL_DIR MODEL_SIZE HS_PREFIX CONFIGS <<< "${MODEL_CFG}"
    TYPE="non"

    echo ""
    echo "####################################################"
    echo "# Model: ${MODEL_NAME} (${MODEL_SIZE})"
    echo "# Model dir: ${MODEL_DIR}"
    echo "# Configs: ${CONFIGS}"
    echo "####################################################"

    # ==================== Baseline ====================
    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] GSM8K (original) - ${MODEL_NAME}"
    echo "=========================================="

    python get_answer_gsm8k.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --test_file "${GSM8K_FILE}" \
        --ans_file "answer_gsm8k" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE}

    echo "[Done] GSM8K original - ${MODEL_NAME}"
    STEP=$((STEP + 1))

    # ==================== Regenerate ====================
    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] GSM8K (regenerate) - ${MODEL_NAME}"
    echo "=========================================="

    python get_answer_regenerate_gsm8k.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --percentage "${PERCENTAGE}" \
        --configs ${CONFIGS} \
        --mask_type "${MASK_TYPE}" \
        --test_file "${GSM8K_FILE}" \
        --ans_file "answer_mdf_gsm8k" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE}

    echo "[Done] GSM8K regenerate - ${MODEL_NAME}"
    STEP=$((STEP + 1))
done

echo ""
echo "=================================================="
echo "All GSM8K benchmarks finished at: $(date)"
echo "=================================================="
