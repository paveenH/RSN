#!/bin/bash

#SBATCH -A MST114558                                 # Account ID
#SBATCH --job-name=judge_confidence                  # Job name
#SBATCH --output=./execution/output_%j.log           # Standard output log
#SBATCH --error=./execution/error_%j.log             # Error output log
#SBATCH --nodes=1                                    # Number of nodes
#SBATCH --ntasks-per-node=1                          # Tasks per node
#SBATCH --gres=gpu:2                                 # Number of GPUs
#SBATCH --cpus-per-task=4                            # Number of CPUs
#SBATCH --time=8:00:00                               # Maximum runtime
#SBATCH --partition=normal                           # Partition
#SBATCH --mail-type=ALL                              # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com            # Email address

# ==================== Confidence Judgment (N5) ====================
# Uses Llama3-70B to judge confidence of GSM8K responses
# Input:  gsm8k/ directory with 3 JSON files (original, positive, negative clean versions)
# Output: gsm8k/confidence_results/
#
# Usage: sbatch run_judge_confidence.sh

# ==================== Configuration ====================
USERNAME="d12922004"
JUDGE_MODEL="/work/${USERNAME}/models/Llama3-70B"
INPUT_DIR="gsm8k"
OUTPUT_DIR="gsm8k/confidence_results"
MAX_NEW_TOKENS=128

# ==================== Paths ====================
WORK_DIR="/work/${USERNAME}/RolePlaying"
CONDA_ENV="roleplaying"

# ==================== Environment ====================
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="
echo "Judge model: ${JUDGE_MODEL}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
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

# ==================== Run ====================
python judge_confidence.py \
    --judge_model_dir "${JUDGE_MODEL}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --files gsm8k_8B_answers_llama3_original_clean.json \
            gsm8k_8B_answers_llama3_positive_clean.json \
            gsm8k_8B_answers_llama3_negative_clean.json

echo ""
echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="