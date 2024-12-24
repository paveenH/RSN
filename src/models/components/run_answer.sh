#!/bin/bash
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:18:42 2024

@author: paveenhuang
"""

# Define the list of tasks
TASKS=(
    "abstract_algebra"
    # Add more tasks as needed
)

# Define the list of model sizes
SIZES=("1B" "3B" "8B")

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the directory where the script is located
cd "$SCRIPT_DIR"

# Number of tasks to execute in parallel (adjust according to your CPU core count)
JOBS=4

# Prepare all combinations of tasks and sizes
COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        COMBINATIONS+=("$TASK" "$SIZE")
    done
done

# Execute using GNU parallel
# Ensure GNU parallel is installed: sudo apt-get install parallel
echo "Starting parallel execution with $JOBS jobs..."
parallel -j "$JOBS" --link python3 get_answer.py {1} {2} ::: "${COMBINATIONS[@]}"

echo "All tasks have been processed successfully."