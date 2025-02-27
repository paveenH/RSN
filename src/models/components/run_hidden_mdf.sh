#!/bin/bash

# Define task list
TASKS=(
    "professional_psychology"
    "public_relations"
    "security_studies"
    "sociology"
    "us_foreign_policy"
    "virology"
    "world_religions"
)

# Define model parameters
MODEL="llama3"    # Your model name, adjustable
SIZE="8B"        # Model size, adjustable
TOP=20            # Select top-K largest hidden states dimensions
ALPHA=1       # Impact factor
START=0          # Starting layer for modification
END=1            # Ending layer for modification

# Define the path to `get_hidden_states_mdf.py`
SCRIPT_PATH="/data2/paveen/RolePlaying/src/models/components/get_hidden_states_mdf.py"

# Loop through all tasks and run the Python script
for TASK in "${TASKS[@]}"; do
    echo "Running task: $TASK with model: $MODEL size: $SIZE"

    # Call Python script
    python "$SCRIPT_PATH" "$TASK $MODEL $SIZE $TOP $ALPHA $START $END"

    echo "Finished processing $TASK"
done

echo "All tasks completed!"