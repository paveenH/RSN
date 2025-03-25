#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024
# Author: paveenhuang

# Define the list of tasks
TASKS=(
"abstract_algebra"
"anatomy"
"astronomy"
"business_ethics"
"clinical_knowledge"
)

# Define the list of model sizes
# SIZES=("0.5B" "3B" "7B")
# MODELS=("qwen2.5")

SIZES=("8B")
MODELS=("llama3")

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the directory where the script is located
cd "$SCRIPT_DIR"

# Prepare all combinations of tasks, models, and sizes
COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SIZE in "${SIZES[@]}"; do
            COMBINATIONS+=("$TASK $MODEL $SIZE")
        done
    done
done

# Debugging: print combinations
echo "Task-Model-Size Combinations to Process:"
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "$COMBINATION"
done

# Execute each combination **sequentially** (removed parallel)
echo "Starting sequential execution..."
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "Processing: $COMBINATION"
    python3 /data2/paveen/RolePlaying/src/models/components/get_answer.py "$COMBINATION"
    if [ $? -ne 0 ]; then
        echo "An error occurred while processing: $COMBINATION"
        exit 1
    fi
done

echo "All tasks have been processed successfully."