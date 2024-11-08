#!/bin/bash

# 定义所有任务
tasks=(
    "high_school_european_history"
    "business_ethics"
    "clinical_knowledge"
    "medical_genetics"
    "high_school_us_history"
    "high_school_physics"
    "high_school_world_history"
    "virology"
    "high_school_microeconomics"
    "econometrics"
    "college_computer_science"
    "high_school_biology"
    "abstract_algebra"
    "professional_accounting"
    "philosophy"
    "professional_medicine"
    "nutrition"
    "global_facts"
    "machine_learning"
    "security_studies"
    "public_relations"
    "professional_psychology"
    "prehistory"
    "anatomy"
    "human_sexuality"
    "college_medicine"
    "high_school_government_and_politics"
    "college_chemistry"
    "logical_fallacies"
    "high_school_geography"
    "elementary_mathematics"
    "human_aging"
    "college_mathematics"
    "high_school_psychology"
    "formal_logic"
    "high_school_statistics"
    "international_law"
    "high_school_mathematics"
    "high_school_computer_science"
    "conceptual_physics"
    "miscellaneous"
    "high_school_chemistry"
    "marketing"
    "professional_law"
    "management"
    "college_physics"
    "jurisprudence"
    "world_religions"
    "sociology"
    "us_foreign_policy"
    "high_school_macroeconomics"
    "computer_security"
    "moral_scenarios"
    "moral_disputes"
    "electrical_engineering"
    "astronomy"
    "college_biology"
)

# 定义集中存储目录
COLLECTED_DIR="./logs/eval/collected_metrics/"
mkdir -p "$COLLECTED_DIR"

# 定义日志文件
LOG_FILE="./logs/eval/collection.log"
touch "$LOG_FILE"

# 定义运行任务的函数
run_task() {
    task=$1
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Running task: $task" | tee -a "$LOG_FILE"
    
    # 获取运行前的目录列表
    before_run=$(ls -td /data2/paveen/RolePlaying/logs/eval/runs/*)
    
    # 运行 Python 脚本
    python3 src/eval.py model=text_otf data=mmlu data.dataset_partial.task="$task"
    if [ $? -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Error running task: $task" | tee -a "$LOG_FILE"
        return 1
    fi

    # 确保所有文件写入完成
    sleep 2

    # 获取运行后的目录列表
    after_run=$(ls -td /data2/paveen/RolePlaying/logs/eval/runs/*)
    
    # 找到新创建的目录
    new_run_dir=$(comm -13 <(echo "$before_run") <(echo "$after_run") | head -n 1)
    
    if [ -z "$new_run_dir" ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] No new run directory found for task: $task" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] New run directory: $new_run_dir" | tee -a "$LOG_FILE"
    
    metrics_path="$new_run_dir/csv/version_0/metrics.csv"
    
    # 检查 metrics.csv 是否存在
    if [ -f "$metrics_path" ]; then
        # 复制并重命名为 [task].csv
        cp "$metrics_path" "$COLLECTED_DIR/$task.csv"
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Saved: $COLLECTED_DIR/$task.csv" | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] metrics.csv not found: $metrics_path" | tee -a "$LOG_FILE"
    fi
}

# 遍历所有任务并运行
for task in "${tasks[@]}"
do
    run_task "$task"
    if [ $? -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Skipping to next task due to error." | tee -a "$LOG_FILE"
    fi
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Completed collection of metrics.csv files." | tee -a "$LOG_FILE"