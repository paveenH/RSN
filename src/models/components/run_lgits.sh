#!/bin/bash
# Example: run multiple tasks and append results to CSV files per role

# Define the list of tasks
# You can also set other parameters if needed:
MODEL="llama3"
SIZE="8B"

# Loop through each task and run the Python script
for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK"
    python3 get_logits.py "$TASK $MODEL $SIZE"
    if [ $? -ne 0 ]; then
        echo "Error processing task: $TASK"
        exit 1
    fi
done

echo "All tasks processed."