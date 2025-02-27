#!/bin/bash

# Define task list
TASKS=(
    "abstract_algebra"
    "anatomy"
    "econometrics"
    "global_facts"
    "jurisprudence"
)

# Define model parameters
MODEL="llama3"    # Your model name, adjustable
SIZE="8B"         # Model size, adjustable
TOP=640            # Select top-K largest hidden states dimensions
ALPHA=1           # Impact factor

# Define the path to `get_hidden_states_mdf.py`
SCRIPT_PATH="/data2/paveen/RolePlaying/src/models/components/get_hidden_states_mdf.py"

# Loop through all tasks
for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK with model: $MODEL, size: $SIZE"
    
    # For each layer from 0 to 31
    for (( layer=0; layer<32; layer++ )); do
        START=$layer
        END=$((layer+1))
        echo "    Processing layer $START to $END"
        
        # Call Python script with the current parameters
        python "$SCRIPT_PATH" "$TASK $MODEL $SIZE $TOP $ALPHA $START $END"
    done
    
    echo "Finished processing task: $TASK"
done

echo "All tasks completed!"