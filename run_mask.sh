#!/bin/bash

MODEL="llama3"
SIZE="8B"
TYPE="non"
PCT="0.5"
BASE="python ./detection/ttest.py"
LOGITS="--logits"

echo "======= Running ALL MASK GENERATION TASKS ======="

# -------------------------
# 1) dense_pca (2 runs)
# -------------------------
for LAYER in "19-20" "32-33" "11-20" "1-33"; do
    echo "[dense_pca] LAYER=$LAYER"
    $BASE \
        --model $MODEL \
        --size $SIZE \
        --type $TYPE \
        --percentage $PCT \
        --mask_type dense_pca \
        --layer $LAYER \
        $LOGITS
done

# -------------------------
# 2) sparse masks (4 types × 2 layer ranges)
# -------------------------
MASKS=("ttest" "ttest_abs" "sparse_pca" "global_sparse_pca")
LAYERS=("11-20" "1-33")

for MASK in "${MASKS[@]}"; do
  for LAYER in "${LAYERS[@]}"; do
      echo "[$MASK] LAYER=$LAYER"
      $BASE \
          --model $MODEL \
          --size $SIZE \
          --type $TYPE \
          --percentage $PCT \
          --mask_type $MASK \
          --layer $LAYER \
          $LOGITS
  done
done

echo "======= ALL 10 TASKS COMPLETE ======="