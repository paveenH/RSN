#!/bin/bash

#SBATCH -A MST114558                            # Account ID
#SBATCH --job-name=rsn_regenerate               # Job name
#SBATCH --output=./execution/output_%j.log      # Standard output log
#SBATCH --error=./execution/error_%j.log        # Error output log
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --gres=gpu:1                            # Number of GPUs (14B needs 1 H100)
#SBATCH --cpus-per-task=4                       # Number of CPUs
#SBATCH --time=03:00:00                         # Maximum runtime
#SBATCH --partition=normal                      # Partition
#SBATCH --mail-type=ALL                         # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com       # Email address

# ==================== Configuration ====================
USERNAME="d12922004"
MODEL_NAME="mistral3"
MODEL_DIR="/work/${USERNAME}/models/Mistral3-14B"
MODEL_SIZE="14B"
TYPE="non"
HS_PREFIX="mistral3"                            # Hidden state folder prefix

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5                                  # Must match the mask file

# Alpha and layer range configurations (format: alpha-start-end)
# Mistral3-14B layer range candidates: [8,24) and [15,24)
CONFIGS="4-8-24 4-15-24 1-1-41"

# Roles
ROLES="{task} expert,non {task} expert"

# Output
ANS_FILE="answer_mdf_mmlue"
SUITE="default"
USE_E="--E"                                      # With E option (A-E)

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

echo "[Running] get_answer_regenerate_logits.py"
echo "Model: ${MODEL_NAME} (${MODEL_DIR})"
echo "Mask: ${MASK_TYPE}, Percentage: ${PERCENTAGE}%"
echo "Configs: ${CONFIGS}"

python get_answer_regenerate_logits.py \
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
    --roles "${ROLES}" \
    ${USE_E}

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
