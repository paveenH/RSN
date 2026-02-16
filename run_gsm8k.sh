#!/bin/bash

#SBATCH -A MST114558                                # Account ID
#SBATCH --job-name=qwen3_gsm8k                   # Job name
#SBATCH --output=./execution/output_%j.log      # Standard output log
#SBATCH --error=./execution/error_%j.log        # Error output log
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --gres=gpu:1                            # Number of GPUs
#SBATCH --cpus-per-task=4                       # Number of CPUs
#SBATCH --time=4:00:00                         # Maximum runtime
#SBATCH --partition=normal                      # Partition
#SBATCH --mail-type=ALL                         # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com       # Email address

# ==================== Configuration ====================
USERNAME="d12922004"
MODEL_NAME="llama3"
MODEL_DIR="/work/${USERNAME}/models/llama3-8B"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"

# Role and template configuration
ROLES="neutral"
SUITE="default"

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Alpha and layer range configurations (format: alpha-start-end)
# Middle layers [11, 20)
CONFIGS="4-11-20 neg4-11-20"

# Generation parameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.0

# GSM8K data file
GSM8K_FILE="benchmark/gsm8k_test.json"

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

# ==================== 1. GSM8K (baseline) ====================
echo ""
echo "=========================================="
echo "[1/2] Running GSM8K (original)"
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

echo "[Done] GSM8K original"

# ==================== 2. GSM8K (regenerate) ====================
echo ""
echo "=========================================="
echo "[2/2] Running GSM8K (regenerate)"
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

echo "[Done] GSM8K regenerate"

echo ""
echo "=================================================="
echo "All benchmarks finished at: $(date)"
echo "=================================================="
