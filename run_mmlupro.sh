#!/bin/bash

#SBATCH -A MST114558                            # Account ID
#SBATCH --job-name=mmlupro_logits               # Job name
#SBATCH --output=./execution/output_%j.log      # Standard output log
#SBATCH --error=./execution/error_%j.log        # Error output log
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --gres=gpu:1                            # Number of GPUs (70B needs 2 H100)
#SBATCH --cpus-per-task=4                       # Number of CPUs
#SBATCH --time=06:00:00                         # Maximum runtime
#SBATCH --partition=normal                      # Partition
#SBATCH --mail-type=ALL                         # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com       # Email address

# ==================== Configuration ====================
USERNAME="d12922004"
MODEL_NAME="llama3"
# MODEL_DIR="/work/${USERNAME}/models/Llama-3.3-70B-Instruction"
MODEL_DIR="/work/${USERNAME}/models/Llama-3.2-1B"
MODEL_SIZE="1B"
TYPE="non"

# Role configuration
ROLES="neutral"

# Output
ANS_FILE="answer_"
SUITE="default"
# No --use_E flag (no E option)

# ==================== Benchmarks ====================
# Files located in: ${BASE_DIR}/benchmark/
# Format: "display_name:filename" (filename without .json extension)
declare -A BENCHMARKS=(
    ["mmlupro"]="benchmark/mmlupro_test"
    ["factor"]="benchmark/factor_mc"
    ["gpqa"]="benchmark/gpqa_train"
    ["arlsat"]="benchmark/arlsat_all"
    ["logiqa"]="benchmark/logiqa_mrc"
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

for NAME in "${!BENCHMARKS[@]}"; do
    TEST_FILE="${BENCHMARKS[$NAME]}"
    echo ""
    echo "=================================================="
    echo "[Running] ${NAME}"
    echo "Test file: ${TEST_FILE}.json"
    echo "Model: ${MODEL_NAME} (${MODEL_DIR})"
    echo "Roles: ${ROLES}"
    echo "=================================================="

    python get_answer_logits_mmlupro.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --test_file "${TEST_FILE}.json" \
        --ans_file "${ANS_FILE}${NAME}" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}"

    echo "[Done] ${NAME}"
done

echo "=================================================="
echo "All benchmarks finished at: $(date)"
echo "=================================================="