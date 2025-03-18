#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:09:34 2025

@author: paveenhuang
"""

import os
import json
from collections import defaultdict
import numpy as np
import re
import csv
# import matplotlib.pyplot as plt


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
model = "llama3_v5"
size = "8B"
top = "20"   # used for layered approach (字符串)
top_org = "20"
start_1 = "0"
end_1   = "31"
start_2 = "10"
end_2   = "31"

# If you want to include the final layer data, they are kept below (not used here)
# last_top_1 = "640"
# last_top_2 = "4096"

answer_layer = "answer_modified_layer_revised"   # layered
answer_original = "answer_honest_revised"         # original
answer_mdf = "answer_modified_revised"            # modifies all layers
answer_index = "answer_modified_index_revised"

base_path = os.getcwd()
original_dir = os.path.join(base_path, model, answer_original)
layer_dir = os.path.join(base_path, model, answer_layer)
mdf_dir = os.path.join(base_path, model, answer_mdf)
index_dir = os.path.join(base_path, model, answer_index)  # 新增：索引数据所在目录

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
        "mdf_none": defaultdict(list),
        "mdf_char": defaultdict(list),
        "layer_none": {
            "0_31": defaultdict(list),
            "0_10": defaultdict(list),
            "10_31": defaultdict(list),
            "31_32_640": defaultdict(list),
            "31_32_4096": defaultdict(list)
        },
        "layer_char": {
            "0_31": defaultdict(list),
            "0_10": defaultdict(list),
            "10_31": defaultdict(list),
            "31_32_640": defaultdict(list),
            "31_32_4096": defaultdict(list)
        },
        "index1_none": defaultdict(list),
        "index1_char": defaultdict(list)
    }

def get_domain(task_name):
    for dom, tasks in categories_map.items():
        if task_name in tasks:
            return dom
    return "Other"

def extract_stats(data, key):
    """
    Extract the data corresponding to the key from 'accuracy' part of the JSON
    and return { "correct": x, "total": y, "E_count": z }
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
    returns (accuracy, e_ratio)
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

# ============ 5) Read Layer: store none/char with segment = 0_31 or 10_31, plus last layer 31_32 for top=640/4096 ============
if os.path.exists(layer_dir):
    for file in os.listdir(layer_dir):
        # Only match files with specified size and answers_{top}, also consider last layer files
        if f"{size}_answers" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        
        # Parse top, start, and end
        top_val = m.group("top")      # e.g. "20", "640", "4096"
        start_val = m.group("start")  # e.g. "0", "10", "31"
        end_val   = m.group("end")    # e.g. "31", "32"
        
        if (start_val is None) or (end_val is None):
            continue
        
        if start_val == "0" and end_val == "31" and top_val == top:
            seg_key = "0_31"
        elif start_val == "10" and end_val == "31" and top_val == top:
            seg_key = "10_31"
        elif start_val == "0" and end_val == "10" and top_val == top:
            seg_key = "0_10"
        elif start_val == "31" and end_val == "32" and top_val in ["640", "4096"]:
            seg_key = f"31_32_{top_val}"
        else:
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
        
        if seg_key in results_dict[domain]["layer_none"]:
            results_dict[domain]["layer_none"][seg_key][task_name].append((acc_none, e_none))
        if seg_key in results_dict[domain]["layer_char"]:
            results_dict[domain]["layer_char"][seg_key][task_name].append((acc_char, e_char))

# ============ 6) Read MDF (none/char) ============
if os.path.exists(mdf_dir):
    for file in os.listdir(mdf_dir):
        if f"{size}_answers_{top_org}" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        
        file_path = os.path.join(mdf_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        none_key = f"none {task_name}"
        char_key = task_name
        
        none_stats = extract_stats(data, none_key)
        char_stats = extract_stats(data, char_key)
        
        acc_none, e_none = compute_acc_e(none_stats)
        acc_char, e_char = compute_acc_e(char_stats)
        
        results_dict[domain]["mdf_none"][task_name].append((acc_none, e_none))
        results_dict[domain]["mdf_char"][task_name].append((acc_char, e_char))

# ============ 7) Read Index data: answer_modified_index_revised ============
# abstract_algebra_8B_answers_20_10_31.json
if os.path.exists(index_dir):
    for file in os.listdir(index_dir):
        if f"{size}_answers" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        
        file_top = m.group("top")      # e.g. "20"
        file_start = m.group("start")  # e.g. "10"
        file_end   = m.group("end")    # e.g. "31"
        
        if file_top != top or file_start != start_2 or file_end != end_2:
            continue
        
        file_path = os.path.join(index_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        none_key = f"none {task_name}"
        char_key = task_name
        
        none_stats = extract_stats(data, none_key)
        char_stats = extract_stats(data, char_key)
        
        acc_none, e_none = compute_acc_e(none_stats)
        acc_char, e_char = compute_acc_e(char_stats)
        
        results_dict[domain]["index1_none"][task_name].append((acc_none, e_none))
        results_dict[domain]["index1_char"][task_name].append((acc_char, e_char))
        

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

# ===================10） Save to CSV ==================

conditions = ["Original", "0-10", "10-31", "index1", "0-31", "31_32_640", "31_32_4096", "MDF"]
csv_rows = []

for domain in results_dict.keys():
    none_values = {
        "Original":   aggregate_condition(results_dict[domain]["original_none"]),
        "0-10":       aggregate_condition(results_dict[domain]["layer_none"]["0_10"]),
        "10-31":      aggregate_condition(results_dict[domain]["layer_none"]["10_31"]),
        "index1":     aggregate_condition(results_dict[domain]["index1_none"]),
        "0-31":       aggregate_condition(results_dict[domain]["layer_none"]["0_31"]),
        "31_32_640":  aggregate_condition(results_dict[domain]["layer_none"]["31_32_640"]),
        "31_32_4096": aggregate_condition(results_dict[domain]["layer_none"]["31_32_4096"]),
        "MDF":        aggregate_condition(results_dict[domain]["mdf_none"]),
        
    }
    expert_values = {
        "Original":   aggregate_condition(results_dict[domain]["original_char"]),
        "0-10":       aggregate_condition(results_dict[domain]["layer_char"]["0_10"]),
        "10-31":      aggregate_condition(results_dict[domain]["layer_char"]["10_31"]),
        "index1":     aggregate_condition(results_dict[domain]["index1_char"]),
        "0-31":       aggregate_condition(results_dict[domain]["layer_char"]["0_31"]),
        "31_32_640":  aggregate_condition(results_dict[domain]["layer_char"]["31_32_640"]),
        "31_32_4096": aggregate_condition(results_dict[domain]["layer_char"]["31_32_4096"]),
        "MDF":        aggregate_condition(results_dict[domain]["mdf_char"]),
    }
    
    for cond in conditions:
        correct, e, incorrect = none_values[cond]
        csv_rows.append({
            "Domain": domain,
            "Group": "None",
            "Condition": cond,
            "Avg_Correct": correct,
            "Avg_E": e,
            "Avg_Incorrect": incorrect
        })
    for cond in conditions:
        correct, e, incorrect = expert_values[cond]
        csv_rows.append({
            "Domain": domain,
            "Group": "Expert",
            "Condition": cond,
            "Avg_Correct": correct,
            "Avg_E": e,
            "Avg_Incorrect": incorrect
        })

csv_output_path = os.path.join(output_dir, f"aggregated_results_{model}_{size}_top_{top}.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Domain", "Group", "Condition", "Avg_Correct", "Avg_E", "Avg_Incorrect"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print(f"Aggregated results saved to CSV: {csv_output_path}")