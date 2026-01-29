#!/bin/bash

#SBATCH -A MST114558                            # Account ID (e.g., GOV113038), view from iservice
#SBATCH --job-name=rsn_logits                   # Job name
#SBATCH --output=./execution/output_%j.log      # Standard output log (%j = job ID)
#SBATCH --error=./execution/error_%j.log        # Error output log
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --gres=gpu:2                            # Number of GPUs (1 H100 is enough for 8B model)
#SBATCH --cpus-per-task=8                       # Number of CPUs
#SBATCH --time=48:00:00                         # Maximum runtime
#SBATCH --partition=normal                      # Partition: dev(2h test) / normal(48h) / normal2(H200)
#SBATCH --mail-type=ALL                         # Email notification: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=paveenhuang@gmail.com       # Email address for job notifications

# ==================== Configuration ====================
# Modify the following variables to match your settings
# export HF_HOME="./cache"                       # Optional: custom HuggingFace cache directory

USERNAME="d12922004"              # Your NCHC account ID
MODEL_NAME="Llama-3.3-70B-Instruct"
MODEL_DIR="/work/d12922004/models/Llama-3.3-70B-Instruct"    # 70B model path
MODEL_SIZE="70B"
TYPE="non"
ROLES="{task} expert,non {task} expert"  # Custom roles (use {task} as placeholder)
ANS_FILE="answer_mmlu"               # Folder name for output answers
SUITE="default"
SAVE_HS="--save"                       # Whether to save hidden states (remove this line to skip)
USE_E="--use_E"                        # Use --use_E flag for five-choice template (A-E); leave empty for four-choice (A-D)

# ==================== Paths ====================
WORK_DIR="/work/${USERNAME}/RolePlaying"
BASE_DIR="${WORK_DIR}/components"       # Root directory for data and output
CONDA_ENV="roleplaying"                # Conda environment name

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

echo "[Running] get_answer_logits.py"
echo "Model: ${MODEL_NAME} (${MODEL_DIR})"
echo "Type: ${TYPE}, Suite: ${SUITE}"

python get_answer_logits.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --size "${MODEL_SIZE}" \
    --roles "${ROLES}" \
    --ans_file "${ANS_FILE}" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    ${SAVE_HS} \
    ${USE_E}

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
