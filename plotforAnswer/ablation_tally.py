#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 answer_ablation2629_revised 文件夹中符合指定条件的文件（例如 business_ethics_8B_answers_20_0_31.json），
并根据任务所属类别（STEM、Humanities、Social Sciences、Other）聚合准确率和 E 比率，
最后将结果保存到 CSV 文件中。
"""

import os
import json
from collections import defaultdict
import numpy as np
import re
import csv

# ============ 1) Define categories and task mappings ============
stem_tasks = [
    "abstract algebra",
    "anatomy",
    "astronomy",
    "college biology",
    "college chemistry",
    "college computer science",
    "college mathematics",
    "college physics",
    "computer security",
    "conceptual physics",
    "electrical engineering",
    "elementary mathematics",
    "high school biology",
    "high school chemistry",
    "high school computer science",
    "high school mathematics",
    "high school physics",
    "high school statistics",
    "machine learning"
]

humanities_tasks = [
    "formal logic",
    "high school european history",
    "high school us history",
    "high school world history",
    "international law",
    "jurisprudence",
    "logical fallacies",
    "moral disputes",
    "moral scenarios",
    "philosophy",
    "prehistory",
    "professional law",
    "world religions"
]

social_sciences_tasks = [
    "econometrics",
    "high school geography",
    "high school government and politics",
    "high school macroeconomics",
    "high school microeconomics",
    "high school psychology",
    "human sexuality",
    "professional psychology",
    "public relations",
    "security studies",
    "sociology",
    "us foreign policy"
]

other_tasks = [
    "business ethics",
    "clinical knowledge",
    "college medicine",
    "global facts",
    "human aging",
    "management",
    "marketing",
    "medical genetics",
    "miscellaneous",
    "nutrition",
    "professional accounting",
    "professional medicine",
    "virology"
]

def get_domain(task_name):
    if task_name in stem_tasks:
        return "STEM"
    elif task_name in humanities_tasks:
        return "Humanities"
    elif task_name in social_sciences_tasks:
        return "Social Sciences"
    else:
        return "Other"

def extract_stats(data, key):
    stats = data.get("accuracy", {}).get(key, {})
    return {
        "correct": stats.get("correct", 0),
        "total": stats.get("total", 0),
        "E_count": stats.get("E_count", 0)
    }

def compute_acc_e(stats):
    total = stats.get("total", 0)
    if total == 0:
        return 0.0, 0.0
    acc = stats.get("correct", 0) / total
    e_ratio = stats.get("E_count", 0) / total
    return acc, e_ratio

# ============ 2) Basic configuration ============
model = "llama3_v3"
size = "8B"
top = "20"  
start = "0"
end = "31"

answer_ablation = "answer_ablation_feq18_revised"
base_path = os.getcwd()
ablation_dir = os.path.join(base_path, model, answer_ablation)

output_dir = os.path.join(base_path, model, "counts")
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(
    r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers"
    r"(?:_(?P<top>\d+))?"
    r"(?:_(?P<start>\d+)_(?P<end>\d+))?"
    r"\.json$"
)

# ============ 3) Data structure ============
domain_keys = ["STEM", "Humanities", "Social Sciences", "Other"]
results_dict = {}
for domain in domain_keys:
    results_dict[domain] = {
        "ablation_none": defaultdict(list),
        "ablation_char": defaultdict(list)
    }

# ============ 4) Read files from ablation_dir ============
if os.path.exists(ablation_dir):
    for file in os.listdir(ablation_dir):
        if f"{size}_answers" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        
        file_path = os.path.join(ablation_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        none_key = f"none {task_name}"
        char_key = task_name
        
        none_stats = extract_stats(data, none_key)
        char_stats = extract_stats(data, char_key)
        acc_none, e_none = compute_acc_e(none_stats)
        acc_char, e_char = compute_acc_e(char_stats)
        
        results_dict[domain]["ablation_none"][task_name].append((acc_none, e_none))
        results_dict[domain]["ablation_char"][task_name].append((acc_char, e_char))
else:
    print(f"Directory {ablation_dir} does not exist!")
    exit(1)

# ============ 5) Aggregation function ============
def aggregate_condition(data_dict):
    all_acc = []
    all_e = []
    for task, values in data_dict.items():
        for (acc, e) in values:
            all_acc.append(acc)
            all_e.append(e)
    if len(all_acc) == 0:
        return 0.0, 0.0, 0.0
    avg_correct = np.mean(all_acc)
    avg_e = np.mean(all_e)
    avg_incorrect = 1 - avg_correct - avg_e
    return avg_correct, avg_e, avg_incorrect

# ============ 6) Save aggregated results to CSV ============
conditions = ["ablation"]

csv_rows = []
for domain in results_dict.keys():
    none_vals = aggregate_condition(results_dict[domain]["ablation_none"])
    expert_vals = aggregate_condition(results_dict[domain]["ablation_char"])
    
    csv_rows.append({
        "Domain": domain,
        "Group": "None",
        "Condition": "ablation",
        "Avg_Correct": none_vals[0],
        "Avg_E": none_vals[1],
        "Avg_Incorrect": none_vals[2]
    })
    csv_rows.append({
        "Domain": domain,
        "Group": "Expert",
        "Condition": "ablation",
        "Avg_Correct": expert_vals[0],
        "Avg_E": expert_vals[1],
        "Avg_Incorrect": expert_vals[2]
    })

csv_output_path = os.path.join(output_dir, f"aggregated_results_{model}_{size}_top_{top}_start_{start}_end_{end}_ablation.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Domain", "Group", "Condition", "Avg_Correct", "Avg_E", "Avg_Incorrect"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print(f"Aggregated results saved to CSV: {csv_output_path}")