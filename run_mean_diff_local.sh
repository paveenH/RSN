#!/bin/bash
# ==================== Mean Diff Calculation Script (Local) ====================
# This script computes mean hidden states from inconsistent samples.
# For local server (not NCHC)
#
# Usage: bash run_mean_diff_local.sh

# ==================== Configuration ====================
MODEL="mistral3"
SIZE="8B"                           # Model size for Mistral3-8B-Reasoning
TYPE="non"
DATA="data2"

# ==================== Paths ====================
WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Run ====================
cd ${WORK_DIR}/mean

echo "=================================================="
echo "Computing mean hidden states (Local Server)"
echo "Model: ${MODEL}, Size: ${SIZE}, Type: ${TYPE}"
echo "=================================================="

python mean_diff.py \
    --model "${MODEL}" \
    --size "${SIZE}" \
    --type "${TYPE}" \
    --base_dir "${BASE_DIR}"

echo "=================================================="
echo "Done!"
echo "=================================================="
