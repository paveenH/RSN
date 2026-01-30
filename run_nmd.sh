#!/bin/bash
# ==================== NMD Mask Generation Script ====================
# This script generates NMD masks from mean hidden states.
# Can be run directly on login node (no GPU required).
#
# Usage: bash run_nmd.sh

# ==================== Configuration ====================
MODEL="llama3"
SIZE="70B"
TYPE="non"
HS_PREFIX="llama3"          # Hidden state folder prefix

# Layer range for mask (adjust based on model)
# Llama-3.3-70B has 81 layers (including embedding), so valid range is [1, 81)
START_LAYER=1
END_LAYER=81

# Percentage of neurons to keep per layer
PERCENTAGE=0.5              # 0.5% of hidden_dim (8192) = ~41 neurons per layer

# Mask type: nmd / random / diff_random / sparse_fv
MASK_TYPE="nmd"

# Random seed (for random/diff_random)
SEED=42

# ==================== Run ====================
cd /work/d12922004/RolePlaying/detection

echo "=================================================="
echo "Generating ${MASK_TYPE} mask"
echo "Model: ${MODEL}, Size: ${SIZE}, Type: ${TYPE}"
echo "Layer range: [${START_LAYER}, ${END_LAYER})"
echo "Percentage: ${PERCENTAGE}%"
echo "=================================================="

python nmd.py \
    --model "${MODEL}" \
    --size "${SIZE}" \
    --type "${TYPE}" \
    --hs "${HS_PREFIX}" \
    --percentage "${PERCENTAGE}" \
    --start_layer "${START_LAYER}" \
    --end_layer "${END_LAYER}" \
    --mask_type "${MASK_TYPE}" \
    --seed "${SEED}" \
    --logits

echo "=================================================="
echo "Done!"
echo "=================================================="
