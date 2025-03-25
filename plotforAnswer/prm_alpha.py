#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 11:25:42 2025

@author: paveenhuang
"""
import os
import json
from collections import defaultdict
import numpy as np
import re
import csv
import matplotlib.pyplot as plt


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

categories_map = {
    "STEM": stem_tasks,
    "Humanities": humanities_tasks,
    "Social Sciences": social_sciences_tasks,
    "Other": other_tasks
}

# ============ 2) Basic configuration ============
model = "llama3_v3"
size = "8B"
top = "20"
start = "11"
end  = "21"

alphas= ["1.0", "1.1", "1.3" ,"1.5", "2.0", "3.0","4.0", "5.0", "6.0"]
# alphas= ["1.0", "2.0", "3.0","4.0", "5.0", "6.0"]
# alphas= ["1.0", "1.1", "1.3" ,"1.5", "2.0", "3.0","4.0"]
# alphas= ["1.0", "1.1", "1.3" ,"1.5", "2.0", "3.0"]



answer_alpha = "answer_modified"   
answer_original = "answer_honest_revised"         # original
answer_mdf = "answer_modified_revised"

base_path = os.getcwd()
original_dir = os.path.join(base_path, model, answer_original)
mdf_dir = os.path.join(base_path, model, answer_mdf)
output_dir = os.path.join(base_path, model, "counts")
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(
    r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers"
    r"(?:_(?P<top>\d+))?"
    r"(?:_(?P<start>\d+)_(?P<end>\d+))?"
    r"\.json$"
)

# ============ 3) Data structure ============
domain_keys = list(categories_map.keys())
results_dict = {}
for domain in domain_keys:
    results_dict[domain] = {
        "original_none": defaultdict(list),
        "original_char": defaultdict(list),
        "alpha_none": {alpha: defaultdict(list) for alpha in alphas},  
        "alpha_expert": {alpha: defaultdict(list) for alpha in alphas}, 
        "none_mdf": defaultdict(list),
        "expert_mdf": defaultdict(list),
    }

def get_domain(task_name):
    for dom, tasks in categories_map.items():
        if task_name in tasks:
            return dom
    print("Error")
    return "Other"

def extract_stats(data, key):
    """
    Extract data from 'accuracy'[key], e.g. "none anatomy"
    Returns { "correct": x, "total": y, "E_count": z }
    """
    stats = data.get("accuracy", {}).get(key, {})
    return {
        "correct": stats.get("correct", 0),
        "total": stats.get("total", 0),
        "E_count": stats.get("E_count", 0)
    }

def compute_acc_e(stats):
    """
    Given stats: { "correct", "total", "E_count" },
    returns (accuracy, e_ratio).
    """
    total = stats.get("total", 0)
    if total == 0:
        return 0.0, 0.0
    acc = stats.get("correct", 0) / total
    e_ratio = stats.get("E_count", 0) / total
    return acc, e_ratio

# ============ 4) Read Original (none/char) ============
if os.path.exists(original_dir):
    for file in os.listdir(original_dir):
        if f"{size}_answers" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        
        file_path = os.path.join(original_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        none_key = f"none {task_name}"
        char_key = task_name
        
        none_stats = extract_stats(data, none_key)
        char_stats = extract_stats(data, char_key)
        acc_none, e_none = compute_acc_e(none_stats)
        acc_char, e_char = compute_acc_e(char_stats)
        
        results_dict[domain]["original_none"][task_name].append((acc_none, e_none))
        results_dict[domain]["original_char"][task_name].append((acc_char, e_char))
        

# ============ 4.5) Read MDF ============
if os.path.exists(mdf_dir):
    for file in os.listdir(mdf_dir):
        if f"{size}_answers" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")   # e.g. "abstract_algebra"
        parsed_size = m.group("size")  # e.g. "8B"
        top_val = m.group("top")        # e.g. "20"
        
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)

        # Check if size matches, top value equals our top, and (start, end) is in our pairs.
        if parsed_size != size:
            continue
        if not top_val or int(top_val) != int(top):
            continue        
        
        file_path = os.path.join(mdf_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        none_key = f"none {task_name}"
        char_key = task_name
        
        none_stats = extract_stats(data, none_key)
        char_stats = extract_stats(data, char_key)
        acc_none, e_none = compute_acc_e(none_stats)
        acc_char, e_char = compute_acc_e(char_stats)
        
        results_dict[domain]["none_mdf"][task_name].append((acc_none, e_none))
        results_dict[domain]["expert_mdf"][task_name].append((acc_char, e_char))
        

# ============ 5) Read Top ============
for alpha in alphas:
    if alpha == "1.0":
        top_tmp = "20"
        alpha_dir = os.path.join(base_path, model, answer_alpha+"_layer_revised")
    else:
        top_tmp = top
        alpha_dir = os.path.join(base_path, model, answer_alpha+f"_alpha{alpha}_revised")
    if not os.path.exists(alpha_dir):
        continue
    
    for file in os.listdir(alpha_dir):
        if f"{size}_answers_{top_tmp}_{start}_{end}" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue

        task_raw = m.group("task")  # e.g. "abstract_algebra"
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        
        # top, start, end
        top_val = m.group("top")      # e.g. "10"
        start_val = m.group("start")  # e.g. "1"
        end_val   = m.group("end")    # e.g. "32"
        
        if start_val != start or end_val != end or top_val != top_tmp:
            continue
        
        file_path = os.path.join(alpha_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        none_key = f"none {task_name}"
        char_key = task_name
        
        none_stats = extract_stats(data, none_key)
        char_stats = extract_stats(data, char_key)
        
        acc_none, e_none = compute_acc_e(none_stats)
        acc_char, e_char = compute_acc_e(char_stats)
        
        results_dict[domain]["alpha_none"][alpha][task_name].append((acc_none, e_none))
        results_dict[domain]["alpha_expert"][alpha][task_name].append((acc_char, e_char))

# ============ 8) Aggregation function ============
def aggregate_condition(data_dict):
    """
    data_dict: dict mapping task -> list of (accuracy, E_ratio) tuples.
    Returns averaged (correct, E, incorrect) across all tasks.
    """
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

# ===================9） Save to CSV ==================
csv_rows = []

for domain in domain_keys:
    # Baseline Original for None
    none_original = aggregate_condition(results_dict[domain]["original_none"])
    csv_rows.append({
        "Domain": domain,
        "Group": "None",
        "Condition": "Original",
        "Avg_Correct": none_original[0],
        "Avg_E": none_original[1],
        "Avg_Incorrect": none_original[2]
    })
    # For each alpha value
    for alpha in alphas:
        condition_name = f"alpha_{alpha}"
        none_agg = aggregate_condition(results_dict[domain]["alpha_none"][alpha])
        csv_rows.append({
            "Domain": domain,
            "Group": "None",
            "Condition": condition_name,
            "Avg_Correct": none_agg[0],
            "Avg_E": none_agg[1],
            "Avg_Incorrect": none_agg[2]
        })
    
    # Baseline Original for Expert
    expert_original = aggregate_condition(results_dict[domain]["original_char"])
    csv_rows.append({
        "Domain": domain,
        "Group": "Expert",
        "Condition": "Original",
        "Avg_Correct": expert_original[0],
        "Avg_E": expert_original[1],
        "Avg_Incorrect": expert_original[2]
    })
    # For each alpha value
    for alpha in alphas:
        condition_name = f"alpha_{alpha}"
        expert_agg = aggregate_condition(results_dict[domain]["alpha_expert"][alpha])
        csv_rows.append({
            "Domain": domain,
            "Group": "Expert",
            "Condition": condition_name,
            "Avg_Correct": expert_agg[0],
            "Avg_E": expert_agg[1],
            "Avg_Incorrect": expert_agg[2]
        })

csv_output_path = os.path.join(output_dir, f"aggregated_results_{model}_{size}_{top}_alpha.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Domain", "Group", "Condition", "Avg_Correct", "Avg_E", "Avg_Incorrect"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print(f"Aggregated results saved to CSV: {csv_output_path}")

# =================10) Plotting Part =================
plt.style.use('seaborn-v0_8-whitegrid')  # paper style

# 1) 构造 baseline（Original）字典
baseline = {}
for row in csv_rows:
    domain = row["Domain"]
    group = row["Group"]  # "None" or "Expert"
    condition = row["Condition"]
    if condition == "Original":
        if domain not in baseline:
            baseline[domain] = {}
        baseline[domain][group] = (
            float(row["Avg_Correct"]),
            float(row["Avg_E"])
        )

# 2) 构造 alpha 的 differences
differences = {}
for row in csv_rows:
    domain = row["Domain"]
    group = row["Group"]
    condition = row["Condition"]
    if condition.startswith("alpha_"):
        alpha_val = float(condition.split("_")[1])  # alpha_X
        acc = float(row["Avg_Correct"])
        e_ratio = float(row["Avg_E"])

        base_acc, base_e = baseline[domain][group]
        diff_acc = acc - base_acc
        diff_e   = e_ratio - base_e

        if domain not in differences:
            differences[domain] = {}
        if group not in differences[domain]:
            differences[domain][group] = {}
        differences[domain][group][alpha_val] = (diff_acc, diff_e)

# 3) 计算 MDF 对 Original 的差值
#    - 先对每个 domain/group 求 MDF 的平均 (acc, e)，再与 baseline 做差
mdf_diff = {"None": [], "Expert": []}  # 用于存储每个domain下MDf与original的差值
for domain in domain_keys:
    # none_mdf
    none_agg = aggregate_condition(results_dict[domain]["none_mdf"])
    base_none_acc, base_none_e = baseline[domain]["None"]
    diff_none_acc = none_agg[0] - base_none_acc
    diff_none_e   = none_agg[1] - base_none_e

    # expert_mdf
    exp_agg = aggregate_condition(results_dict[domain]["expert_mdf"])
    base_exp_acc, base_exp_e = baseline[domain]["Expert"]
    diff_exp_acc = exp_agg[0] - base_exp_acc
    diff_exp_e   = exp_agg[1] - base_exp_e

    # 加到列表里
    mdf_diff["None"].append((diff_none_acc, diff_none_e))
    mdf_diff["Expert"].append((diff_exp_acc, diff_exp_e))

# 3.1) 再对所有 domain 做平均，得到一条水平参考线
mdf_reference = {}
for group in ["None", "Expert"]:
    diffs = mdf_diff[group]
    if diffs:
        avg_acc_diff = np.mean([d[0] for d in diffs])
        avg_e_diff   = np.mean([d[1] for d in diffs])
    else:
        avg_acc_diff, avg_e_diff = 0.0, 0.0
    mdf_reference[group] = (avg_acc_diff, avg_e_diff)


# 4) 计算 alpha 的 overall difference
overall = {"None": {}, "Expert": {}}
for group in ["None", "Expert"]:
    for alpha_str in alphas:
        alpha_val = float(alpha_str)
        acc_list = []
        e_list = []
        for dom in differences:
            if group in differences[dom] and alpha_val in differences[dom][group]:
                diff_acc, diff_e = differences[dom][group][alpha_val]
                acc_list.append(diff_acc)
                e_list.append(diff_e)
        if acc_list:
            overall[group][alpha_val] = (
                np.mean(acc_list),
                np.mean(e_list)
            )

# 5) 画图
for group in ["None", "Expert"]:
    plt.figure(figsize=(10, 6))

    # 5.1) alpha 各 domain 的曲线
    for domain in differences:
        if group in differences[domain]:
            domain_alphas = sorted(differences[domain][group].keys())
            domain_diff_acc = [differences[domain][group][a][0] for a in domain_alphas]
            domain_diff_e   = [differences[domain][group][a][1] for a in domain_alphas]
            plt.plot(domain_alphas, domain_diff_acc, marker='o', linestyle='--', alpha=0.4, label=f"{domain} Acc")
            plt.plot(domain_alphas, domain_diff_e, marker='s', linestyle=':', alpha=0.4, label=f"{domain} E")

    # 5.2) alpha overall 平均曲线（粗线）
    overall_alphas = sorted(overall[group].keys())
    overall_acc = [overall[group][a][0] for a in overall_alphas]
    overall_e   = [overall[group][a][1] for a in overall_alphas]
    plt.plot(overall_alphas, overall_acc, marker='o', color='black', linewidth=2.5, label="Overall Acc Mean")
    plt.plot(overall_alphas, overall_e, marker='s', color='red',   linewidth=2.5, label="Overall E Mean")

    # 5.3) MDF 水平参考线 (两条：Acc 与 E)，使用 axhline
    #     也可以只画一条参考线(比如 Acc)，看需求
    ref_acc = mdf_reference[group][0]
    ref_e   = mdf_reference[group][1]
    plt.axhline(y=ref_acc, color='blue', linestyle='--', label="MDF Acc Reference")
    plt.axhline(y=ref_e,   color='green', linestyle='--', label="MDF E Reference")

    plt.xlabel("Alpha Value", fontsize=12, fontweight="bold")
    plt.ylabel("Difference (Modified - Original)", fontsize=12, fontweight="bold")
    plt.title(f"{group}: Difference in Accuracy & E Ratio vs. Original (Layer {start}-{int(end)-1}, top={top})", 
              fontsize=14, fontweight="bold")
    plt.xticks(overall_alphas)
    plt.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.show()
