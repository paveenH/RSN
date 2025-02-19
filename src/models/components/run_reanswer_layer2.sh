#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024
# Author: paveenhuang

# Define the list of tasks (only 5 tasks)
TASKS=(
  "anatomy"
  "abstract_algebra"
  "global_facts"
  "econometrics"
  "jurisprudence"
)

# Define the list of model sizes, models, tops, and alphas
SIZES=("8B")
MODELS=("llama3")
TOPS=("4096")
ALPHAS=("1")

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the directory where the script is located
cd "$SCRIPT_DIR"

# Number of parallel jobs (adjust according to your CPU core count)
JOBS=1

# Prepare all combinations of tasks, models, sizes, tops, alphas, and each layer (0-31)
COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
      for TOP in "${TOPS[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
          for (( layer=0; layer<32; layer++ )); do
            # For each layer, set start = current layer and end = current layer + 1
            COMBINATIONS+=("$TASK $MODEL $SIZE $TOP $ALPHA $layer $((layer+1))")
          done
        done
      done
    done
  done
done

# Debugging: print combinations
echo "Task-Model-Size-Top-Alpha-Layer Combinations to Process:"
for COMBINATION in "${COMBINATIONS[@]}"; do
  echo "$COMBINATION"
done

# Execute using GNU parallel
# Ensure GNU parallel is installed: sudo apt-get install parallel
echo "Starting parallel execution with $JOBS jobs..."
parallel -j "$JOBS" python3 /data2/paveen/RolePlaying/src/models/components/get_answer_regenerate_layer.py ::: "${COMBINATIONS[@]}"

# Check if parallel execution was successful
if [ $? -eq 0 ]; then
  echo "All tasks have been processed successfully."
else
  echo "An error occurred during parallel execution."
  exit 1
fi