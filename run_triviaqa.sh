#!/bin/bash

#SBATCH -A MST114558                                # Account ID
#SBATCH --job-name=triviaqa_bench                 # Job name
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

# Role and template configuration
ROLES="neutral"
SUITE="default"

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Generation parameters
MAX_NEW_TOKENS=64
TEMPERATURE=0.0

# TriviaQA data file
TRIVIAQA_FILE="benchmark/triviaqa_validation.json"

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
echo "Task: TriviaQA"
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
    echo "[${STEP}/${TOTAL}] TriviaQA (original) - ${MODEL_NAME}"
    echo "=========================================="

    python get_answer_triviaqa.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --test_file "${TRIVIAQA_FILE}" \
        --ans_file "answer_triviaqa" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE}

    echo "[Done] TriviaQA original - ${MODEL_NAME}"
    STEP=$((STEP + 1))

    # ==================== Regenerate ====================
    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] TriviaQA (regenerate) - ${MODEL_NAME}"
    echo "=========================================="

    python get_answer_regenerate_triviaqa.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --percentage "${PERCENTAGE}" \
        --configs ${CONFIGS} \
        --mask_type "${MASK_TYPE}" \
        --test_file "${TRIVIAQA_FILE}" \
        --ans_file "answer_mdf_triviaqa" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE}

    echo "[Done] TriviaQA regenerate - ${MODEL_NAME}"
    STEP=$((STEP + 1))
done

echo ""
echo "=================================================="
echo "All TriviaQA benchmarks finished at: $(date)"
echo "=================================================="
