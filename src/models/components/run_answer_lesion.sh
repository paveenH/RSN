#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024
# Author: paveenhuang

# Define model and size parameters
MODEL="llama3"
SIZE="8B"
START=1
END=31

# Define multiple neuron index lists (comma-separated if there are multiple, here only a single number is set)
NEURON_LISTS=("4055" "2692" "2628")

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Automatically set number of parallel jobs based on the number of CPU cores
JOBS=$(nproc)  # This will set JOBS to the number of available CPU cores

# Prepare all combinations: Each combination is a string with 5 parts: MODEL, SIZE, START, END, and neuron_indices.
COMBINATIONS=()
for INDEX in "${NEURON_LISTS[@]}"; do
    COMBINATIONS+=("$MODEL $SIZE $START $END $INDEX")
done

# Debug: Print combinations
echo "Model-Size-Start-End-NeuronIndices combinations to process:"
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "$COMBINATION"
done

# Ensure parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel could not be found. Please install it."
    exit 1
fi

# Execute using GNU parallel, calling the get_answer_lesion_sample.py script.
echo "Starting parallel execution with $JOBS job(s)..."
parallel -j "$JOBS" python3 /data2/paveen/RolePlaying/src/models/components/get_answer_lesion_sample.py ::: "${COMBINATIONS[@]}" > process.log 2>&1

if [ $? -eq 0 ]; then
    echo "All tasks processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi