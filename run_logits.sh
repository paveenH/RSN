#!/bin/bash

#SBATCH -A <YOUR_ACCOUNT_ID>            # Account ID (e.g., GOV113038), view from iservice
#SBATCH --job-name=rsn_logits           # Job name
#SBATCH --output=output_%j.log          # Standard output log (%j = job ID)
#SBATCH --error=output_%j.log           # Error output (merged to same file)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --gres=gpu:1                    # Number of GPUs (1 H100 is enough for 8B model)
#SBATCH --cpus-per-task=8               # Number of CPUs
#SBATCH --time=48:00:00                 # Maximum runtime
#SBATCH --partition=normal              # Partition: dev(2h test) / normal(48h) / normal2(H200)

# ==================== Configuration ====================
# Modify the following variables to match your settings
USERNAME="<YOUR_USERNAME>"              # Your NCHC account ID
MODEL_NAME="llama3_base"
MODEL_DIR="meta-llama/Llama-3.1-8B"    # HuggingFace model path
MODEL_SIZE="8B"
TYPE="non"
ANS_FILE="answer_logits"               # Folder name for output answers
DATA="data2"                            # Data directory identifier
SUITE="default"
SAVE_HS="--save"                        # Whether to save hidden states (remove this line to skip)

# ==================== Paths ====================
WORK_DIR="/work/${USERNAME}/RolePlaying"
CODE_DIR="${WORK_DIR}/code"
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
cd ${CODE_DIR}

echo "[Running] get_answer_logits.py"
echo "Model: ${MODEL_NAME} (${MODEL_DIR})"
echo "Type: ${TYPE}, Suite: ${SUITE}"

python get_answer_logits.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --ans_file "${ANS_FILE}" \
    --data "${DATA}" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    ${SAVE_HS}

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
