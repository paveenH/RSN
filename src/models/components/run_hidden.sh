#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024
# Author: paveenhuang

# === Description ===
# This script iterates through predefined lists of tasks and model sizes,
# executing the Python script `get_answer.py` for each task-size combination.
# It utilizes GNU Parallel to run multiple jobs in parallel for efficiency.

# === Define the list of tasks ===
TASKS=(
    "abstract_algebra"
    "anatomy"
    "astronomy"
)

# === Define the list of model sizes ===
SIZES=("1B" "3B" "8B")

# === Get the directory where the script is located ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === Change to the script's directory ===
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

# === Define the Python script's path ===
PYTHON_SCRIPT="get_answer.py"

# === Check if the Python script exists ===
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in $SCRIPT_DIR."
    exit 1
fi

# === Check if Python3 is installed ===
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 could not be found. Please install Python3."
    exit 1
fi

# === Define the number of parallel jobs ===
JOBS=1  # Adjust this number based on your CPU core count

# === Prepare all combinations of tasks and sizes ===
COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        COMBINATIONS+=("$TASK" "$SIZE")
    done
done

# === Debugging: Print combinations ===
echo "Task-Size Combinations to Process:"
for ((i=0; i<${#COMBINATIONS[@]}; i+=2)); do
    TASK_NAME="${COMBINATIONS[i]}"
    SIZE_NAME="${COMBINATIONS[i+1]}"
    echo "  Task: $TASK_NAME, Size: $SIZE_NAME"
done

# === Execute the Python script using GNU Parallel ===
# Ensure GNU Parallel is installed: sudo apt-get install parallel
echo "Starting parallel execution with $JOBS jobs..."

# Run the Python script for each task-size combination in parallel
parallel -j "$JOBS" python3 "$PYTHON_SCRIPT" {1} {2} ::: "${COMBINATIONS[@]}"

# === Check if parallel execution was successful ===
if [ $? -eq 0 ]; then
    echo "All tasks have been processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi