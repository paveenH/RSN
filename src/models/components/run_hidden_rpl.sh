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

JOBS=2  # Number of tasks to run in parallel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print all the commands that will be executed
echo "Task-Model-Size-Start-End Combinations to Process:"
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SIZE in "${SIZES[@]}"; do
            for PAIR in "${START_END_PAIRS[@]}"; do
                START_LAYER=$(echo $PAIR | awk '{print $1}')
                END_LAYER=$(echo $PAIR | awk '{print $2}')
                echo "python3 get_hidden_states_rpl.py $TASK $SIZE $MODEL --start $START_LAYER --end $END_LAYER"
            done
        done
    done
done

# Use GNU parallel to execute tasks in parallel
echo "Starting parallel execution with $JOBS jobs..."
parallel -j "$JOBS" python3 get_hidden_states_rpl.py {1} {2} {3} --start {4} --end {5} ::: "${TASKS[@]}" ::: "${SIZES[@]}" ::: "${MODELS[@]}" ::: $(seq $START $((END-1))) ::: $(seq 1 $END)

# Check if parallel execution was successful
if [ $? -eq 0 ]; then
    echo "All tasks have been processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi