#!/bin/bash

MODEL="llama3"
SIZE="8B"
TYPE="non"
PCT="0.5"
BASE="python ./detection/ttest.py"
LOGITS="--logits"

echo "======= Running ALL MASK GENERATION TASKS ======="

: <<'EOF'

# ------------------------------------------
# 1) dense_pca (2 runs: 11–20, 1–33)
# ------------------------------------------
DENSE_LAYERS=("11-20" "1-33")

for LAYER in "${DENSE_LAYERS[@]}"; do
    echo "[dense_pca] LAYER=$LAYER"
    $BASE \
        --model $MODEL \
        --size $SIZE \
        --type $TYPE \
        --percentage "100" \
        --mask_type dense_pca \
        --layer $LAYER \
        $LOGITS
done

EOF
# -------------------------------------------------------
# 2) sparse masks (ttest, ttest_abs, sparse_pca, pca_selection)
#    each run with 2 layer ranges (11–20, 1–33)
# -------------------------------------------------------
MASKS=("ttest_layer" "ttest_layer_abs")
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

echo "======= ALL MASK GENERATION COMPLETE ======="