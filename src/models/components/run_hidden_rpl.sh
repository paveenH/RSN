#!/bin/bash
# Created on $(date)
# Author: paveenhuang
# This script iterates over different tasks, model sizes, and layer pairs 
# to call get_hidden_states_rpl.py, replacing hidden states in a single-layer range.

# Define tasks, model sizes, and models
TASKS=(
    "abstract_algebra"
    "anatomy"
    "global_facts"
    "econometrics"
    "jurisprudence"
)

SIZES=("8B")
MODELS=("llama3")

START=0
END=31

# Define start-end layer pairs: [(0-1), (1-2), ..., (30-31)]
START_END_PAIRS=()
for ((i=START; i<END; i++)); do
    START_END_PAIRS+=("$i $((i+1))")
done

JOBS=2  # Number of parallel jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the combinations: TASK MODEL SIZE start end
COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SIZE in "${SIZES[@]}"; do
            for PAIR in "${START_END_PAIRS[@]}"; do
                COMBINATIONS+=("$TASK $SIZE $MODEL $PAIR")
            done
        done
    done
done

# Print combinations for debugging
echo "Task-Model-Size-Start-End Combinations to Process:"
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "$COMBINATION"
done

# Execute using GNU parallel for better efficiency
echo "Starting parallel execution with $JOBS jobs..."
parallel -j "$JOBS" python3 /data2/paveen/RolePlaying/src/models/components/get_hidden_states_rpl.py ::: "${COMBINATIONS[@]}"

# Check if parallel execution is successful
if [ $? -eq 0 ]; then
    echo "All tasks have been processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi