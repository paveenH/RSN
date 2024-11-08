#!/bin/bash

# 定义任务列表
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

# 定义日志保存目录
COLLECTED_DIR="./collected_metrics/"
mkdir -p "$COLLECTED_DIR"

# 定义运行任务的函数
run_task() {
    task=$1
    echo "正在运行任务: $task"

    # 运行评估，并为每个任务指定单独的输出目录
    python3 src/eval.py model=text_otf data=mmlu data.dataset_partial.task="$task" 

    # 定义 metrics.csv 的路径
    metrics_path="./logs/eval/runs/$task/csv/version_0/metrics.csv"

    # 检查 metrics.csv 是否存在
    if [ -f "$metrics_path" ]; then
        # 复制并重命名为 [task].csv
        cp "$metrics_path" "$COLLECTED_DIR/$task.csv"
        echo "已保存: $COLLECTED_DIR/$task.csv"
    else
        echo "未找到 metrics.csv: $metrics_path" >> "$COLLECTED_DIR/error.log"
    fi
}

# 逐个运行任务
for task in "${tasks[@]}"
do
    run_task "$task"
done

echo "所有任务已完成并收集 metrics.csv 文件。"