#!/bin/bash

# Define the task name
TASK="marketing"

# Number of repetitions
REPEATS=10

# Define collection path
COLLECTED_DIR="./logs/collected_metrics_${TASK}/"
mkdir -p "$COLLECTED_DIR"

# Defining log files
LOG_FILE="./logs/eval/${TASK}_repeats.log"
touch "$LOG_FILE"

# Define the function to run the task
run_task() {
    run_index=$1
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Running task: ${TASK}, iteration: ${run_index}" | tee -a "$LOG_FILE"
    
    # Get a directory listing before running
    before_run=$(ls -td /data2/paveen/RolePlaying/logs/eval/runs/*)
    
    # Run Python script
    python3 src/eval.py model=text_otf data=mmlu data.dataset_partial.task="$TASK"
    if [ $? -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Error running task: ${TASK}, iteration: ${run_index}" | tee -a "$LOG_FILE"
        return 1
    fi

    # Make sure all files are written
    sleep 2

    # Get the directory listing after running
    after_run=$(ls -td /data2/paveen/RolePlaying/logs/eval/runs/*)
    
    # Find the newly created directory
    new_run_dir=$(comm -13 <(echo "$before_run") <(echo "$after_run") | head -n 1)
    
    if [ -z "$new_run_dir" ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] No new run directory found for task: ${TASK}, iteration: ${run_index}" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] New run directory: $new_run_dir" | tee -a "$LOG_FILE"
    
    metrics_path="$new_run_dir/csv/version_0/metrics.csv"

    # Check if metrics.csv exists
    if [ -f "$metrics_path" ]; then
        # Copy and rename to [task]_[iteration].csv
        cp "$metrics_path" "$COLLECTED_DIR/${TASK}_${run_index}.csv"
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Saved: $COLLECTED_DIR/${TASK}_${run_index}.csv" | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] metrics.csv not found: $metrics_path" | tee -a "$LOG_FILE"
    fi

    # Copy all files from the run directory into a subfolder
    subfolder="$COLLECTED_DIR/run_${run_index}"
    mkdir -p "$subfolder"
    cp -r "$new_run_dir"/* "$subfolder"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] All files copied to: $subfolder" | tee -a "$LOG_FILE"
}

# Iterate and run the task multiple times
for i in $(seq 1 $REPEATS)
do
    run_task "$i"
    if [ $? -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Skipping to next iteration due to error." | tee -a "$LOG_FILE"
    fi
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Completed ${REPEATS} repetitions for task: ${TASK}." | tee -a "$LOG_FILE"
