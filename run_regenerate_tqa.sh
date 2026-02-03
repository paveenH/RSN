#!/bin/bash

#SBATCH -A MST114558                            # Account ID
#SBATCH --job-name=regen_tqa                    # Job name
#SBATCH --output=./execution/output_%j.log      # Standard output log
#SBATCH --error=./execution/error_%j.log        # Error output log
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --gres=gpu:2                            # Number of GPUs (70B needs 2 H100)
#SBATCH --cpus-per-task=8                       # Number of CPUs
#SBATCH --time=06:00:00                         # Maximum runtime
#SBATCH --partition=normal                      # Partition
#SBATCH --mail-type=ALL                         # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com       # Email address

# ==================== Configuration ====================
USERNAME="d12922004"
MODEL_NAME="llama3"
MODEL_DIR="/work/${USERNAME}/models/Llama-3.3-70B-Instruct"
MODEL_SIZE="70B"
TYPE="non"
HS_PREFIX="llama3"                              # Hidden state folder prefix

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5                                  # Must match the mask file

# Alpha and layer range configurations (format: alpha-start-end)
CONFIGS="4-17-30 neg4-17-30"

# Role configuration
ROLES="neutral"

# Output
ANS_FILE="answer_mdf_tqa"
SUITE="default"
# No --use_E flag (no E option)

# TruthfulQA modes: mc1 and mc2
MODES=("mc1" "mc2")

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

# Load miniconda
ml load miniconda3

# Activate conda environment
conda activate ${CONDA_ENV}

# Check environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi

# ==================== Run ====================
cd ${WORK_DIR}

for MODE in "${MODES[@]}"; do
    echo ""
    echo "=================================================="
    echo "[Running] TruthfulQA ${MODE} with regenerate"
    echo "Model: ${MODEL_NAME} (${MODEL_DIR})"
    echo "Mask: ${MASK_TYPE}, Percentage: ${PERCENTAGE}%"
    echo "Configs: ${CONFIGS}"
    echo "Roles: ${ROLES}"
    echo "=================================================="

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
        --ans_file "${ANS_FILE}" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}"

    echo "[Done] TruthfulQA ${MODE}"
done

echo "=================================================="
echo "All TruthfulQA modes finished at: $(date)"
echo "=================================================="
