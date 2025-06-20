#!/bin/bash
# Created on $(date)
# Author: paveenhuang
# This script iterates over different tasks, model sizes, and layer pairs 
# to call get_hidden_states_rpl.py, replacing hidden states in a single-layer range.

# Define tasks, model sizes, and model types
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

# Generate start-end layer pairs: [(0-1), (1-2), ..., (30-31)]
START_END_PAIRS=()
for ((i=START; i<END; i++)); do
    START_END_PAIRS+=("$i $((i+1))")
done

JOBS=1  # Number of parallel jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the full command lines for each combination
CMD_LIST=()
for TASK in "${TASKS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            for PAIR in "${START_END_PAIRS[@]}"; do
                START_LAYER=$(echo $PAIR | awk '{print $1}')
                END_LAYER=$(echo $PAIR | awk '{print $2}')
                CMD="python3 get_hidden_states_rpl.py $TASK $SIZE $MODEL --start $START_LAYER --end $END_LAYER"
                CMD_LIST+=("$CMD")
            done
        done
    done
done

# Print all commands for debugging
echo "The following commands will be executed:"
for CMD in "${CMD_LIST[@]}"; do
    echo "$CMD"
done

# Execute the commands in parallel using GNU parallel
echo "Starting parallel execution with $JOBS jobs..."
printf "%s\n" "${CMD_LIST[@]}" | parallel -j "$JOBS"

# Check if parallel execution was successful
if [ $? -eq 0 ]; then
    echo "All tasks have been processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi