#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:10:25 2025

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
top_list = ["1", "3", "5", "10", "15", "20", "25", "30", "40", "50"]
start = "1"
end  = "32"

answer_layer = "answer_modified_layer_revised"   # layered
answer_original = "answer_honest_revised"         # original


base_path = os.getcwd()
original_dir = os.path.join(base_path, model, answer_original)
layer_dir = os.path.join(base_path, model, answer_layer)

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
        "top_none": {top_val: defaultdict(list) for top_val in top_list},   # "1": {...}, "3": {...}, ...
        "top_expert": {top_val: defaultdict(list) for top_val in top_list}, # 同上
    }

def get_domain(task_name):
    for dom, tasks in categories_map.items():
        if task_name in tasks:
            return dom
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

# ============ 5) Read Top ============
if os.path.exists(layer_dir):
    for top in top_list:
        for file in os.listdir(layer_dir):
            if f"{size}_answers_{top}_{start}_{end}" not in file:
                continue
            m = pattern.match(file)
            if not m:
                continue

            task_raw = m.group("task")  # e.g. "abstract_algebra"
            task_name = task_raw.replace('_', ' ')
            domain = get_domain(task_name)
            
            top_val = m.group("top")      # e.g. "10"
            start_val = m.group("start")  # e.g. "1"
            end_val   = m.group("end")    # e.g. "32"
            
            if start_val != "1" or end_val != "32" or (top_val not in top_list):
                continue
            
            file_path = os.path.join(layer_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            none_key = f"none {task_name}"
            char_key = task_name
            
            none_stats = extract_stats(data, none_key)
            char_stats = extract_stats(data, char_key)
            
            acc_none, e_none = compute_acc_e(none_stats)
            acc_char, e_char = compute_acc_e(char_stats)
            
            results_dict[domain]["top_none"][top_val][task_name].append((acc_none, e_none))
            results_dict[domain]["top_expert"][top_val][task_name].append((acc_char, e_char))


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

for domain in results_dict.keys():
    # None
    none_original = aggregate_condition(results_dict[domain]["original_none"])
    csv_rows.append({
        "Domain": domain,
        "Group": "None",
        "Condition": "Original",
        "Avg_Correct": none_original[0],
        "Avg_E": none_original[1],
        "Avg_Incorrect": none_original[2]
    })
    for top_val in top_list:
        none_agg = aggregate_condition(results_dict[domain]["top_none"][top_val])
        csv_rows.append({
            "Domain": domain,
            "Group": "None",
            "Condition": f"top_{top_val}",
            "Avg_Correct": none_agg[0],
            "Avg_E": none_agg[1],
            "Avg_Incorrect": none_agg[2]
        })
    
    # Expert
    expert_original = aggregate_condition(results_dict[domain]["original_char"])
    csv_rows.append({
        "Domain": domain,
        "Group": "Expert",
        "Condition": "Original",
        "Avg_Correct": expert_original[0],
        "Avg_E": expert_original[1],
        "Avg_Incorrect": expert_original[2]
    })
    for top_val in top_list:
        expert_agg = aggregate_condition(results_dict[domain]["top_expert"][top_val])
        csv_rows.append({
            "Domain": domain,
            "Group": "Expert",
            "Condition": f"top_{top_val}",
            "Avg_Correct": expert_agg[0],
            "Avg_E": expert_agg[1],
            "Avg_Incorrect": expert_agg[2]
        })

csv_output_path = os.path.join(output_dir, f"aggregated_results_{model}_{size}_top.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Domain", "Group", "Condition", "Avg_Correct", "Avg_E", "Avg_Incorrect"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print(f"Aggregated results saved to CSV: {csv_output_path}")


# ================= 10）Plotting Part =================
plt.style.use('seaborn-v0_8-whitegrid')  # Set to paper style
# plt.style.use('seaborn-v0_8-paper')  # Set to paper style
baseline = {}
for row in csv_rows:
    domain = row["Domain"]
    group = row["Group"]  # "None" or "Expert"
    condition = row["Condition"]
    if condition == "Original":
        if domain not in baseline:
            baseline[domain] = {}
        baseline[domain][group] = (float(row["Avg_Correct"]), float(row["Avg_E"]))

# differences[domain][group][top_val] = (diff_acc, diff_E)
differences = {}
for row in csv_rows:
    domain = row["Domain"]
    group = row["Group"]
    condition = row["Condition"]
    if condition.startswith("top_"):
        top_val = int(condition.split("_")[1])
        acc = float(row["Avg_Correct"])
        e_ratio = float(row["Avg_E"])
        base_acc, base_e = baseline[domain][group]
        diff_acc = acc - base_acc
        diff_e = e_ratio - base_e
        if domain not in differences:
            differences[domain] = {}
        if group not in differences[domain]:
            differences[domain][group] = {}
        differences[domain][group][top_val] = (diff_acc, diff_e)

# Compute overall averages: for each group, calculate the mean by top value across all domains
overall = {"None": {}, "Expert": {}}
for group in ["None", "Expert"]:
    for top_str in top_list:
        top_val = int(top_str)
        acc_list = []
        e_list = []
        for domain in differences:
            if group in differences[domain] and top_val in differences[domain][group]:
                diff_acc, diff_e = differences[domain][group][top_val]
                acc_list.append(diff_acc)
                e_list.append(diff_e)
        if acc_list:
            overall[group][top_val] = (np.mean(acc_list), np.mean(e_list))

# Plot the graphs
# For each Group (None and Expert), plot a graph containing the curves for each domain and the overall mean curve
for group in ["None", "Expert"]:
    plt.figure(figsize=(10,6))
    
    # Plot curves for each domain (Accuracy and E ratio)
    for domain in differences:
        if group in differences[domain]:
            domain_top_vals = sorted(differences[domain][group].keys())
            domain_diff_acc = [differences[domain][group][tv][0] for tv in domain_top_vals]
            domain_diff_e = [differences[domain][group][tv][1] for tv in domain_top_vals]
            # Use semi-transparent lines to show the trend of each domain
            plt.plot(domain_top_vals, domain_diff_acc, marker='o', linestyle='--', alpha=0.4, label=f"{domain}Acc")
            plt.plot(domain_top_vals, domain_diff_e, marker='s', linestyle=':', alpha=0.4, label=f"{domain} E")
    
    # Plot overall mean curves (use thicker lines)
    overall_top_vals = sorted(overall[group].keys())
    overall_diff_acc = [overall[group][tv][0] for tv in overall_top_vals]
    overall_diff_e = [overall[group][tv][1] for tv in overall_top_vals]
    plt.plot(overall_top_vals, overall_diff_acc, marker='o', color='black', linewidth=2.5, label="Overall Acc Mean")
    plt.plot(overall_top_vals, overall_diff_e, marker='s', color='red', linewidth=2.5, label="Overall E Mean")
    
    plt.xlabel("Top Value", fontsize=12, fontweight="bold")
    plt.ylabel("Difference (Modified - Original)", fontsize=12, fontweight="bold")
    plt.title(f"{group}: Difference in Accuracy & E Ratio (llama3-8B layer[{start}, {int(end)-1}])", fontsize=14, fontweight="bold")
    plt.xticks(overall_top_vals)
    plt.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.show()

# If you need to save the figure to a file, you can use plt.savefig(), for example:
# plt.savefig(os.path.join(output_dir, f"{model}_{size}_{group}_differences.png"), dpi=300)