#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:55:23 2025

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

top = 20
start_end_pairs = [("1","11"), ("11","21"), ("21","32"), ("11","32"), ("1","32"),  ("1", "33") ]

answer_layer = "answer_modified_layer_revised"   # layered
answer_mdf = "answer_modified_revised"
answer_original = "answer_honest_revised"         # original


base_path = os.getcwd()
original_dir = os.path.join(base_path, model, answer_original)
mdf_dir = os.path.join(base_path, model, answer_mdf)
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
        "none": {
            (s, e): defaultdict(list)
            for (s, e) in start_end_pairs
        },
        "expert": {
            (s, e): defaultdict(list)
            for (s, e) in start_end_pairs
        }
    }

def get_domain(task_name):
    for dom, tasks in categories_map.items():
        if task_name in tasks:
            return dom
    print("ERROR in Domain")
    return "Other"

def extract_stats(data, key):
    """
    Extract data from 'accuracy'[key].
    Returns { "correct": x, "total": y, "E_count": z }.
    """
    stats = data.get("accuracy", {}).get(key, {})
    return {
        "correct": stats.get("correct", 0),
        "total": stats.get("total", 0),
        "E_count": stats.get("E_count", 0)
    }

def compute_acc_e(stats):
    """
    Returns (accuracy, e_ratio).
    """
    total = stats.get("total", 0)
    if total == 0:
        return 0.0, 0.0
    acc = stats["correct"] / total
    e_ratio = stats["E_count"] / total
    return acc, e_ratio


# ============ 4) Read Original (none/char) ============
if os.path.exists(original_dir):
    for file in os.listdir(original_dir):
        if f"{size}_answers" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")     # e.g. "abstract_algebra"
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

        # Check if size matches, top value equals our top, and (start, end) is in our pairs.
        if parsed_size != size:
            continue
        if not top_val or int(top_val) != top:
            continue

        task_raw = m.group("task")     # e.g. "abstract_algebra"
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
        
        results_dict[domain]["none"][("1", "33")][task_name].append((acc_none, e_none))
        results_dict[domain]["expert"][("1", "33")][task_name].append((acc_char, e_char))


# ============ 5) Read Top ============
if os.path.exists(layer_dir):
    for file in os.listdir(layer_dir):
        m = pattern.match(file)
        if not m:
            continue
        
        task_raw = m.group("task")   # e.g. "abstract_algebra"
        parsed_size = m.group("size")  # e.g. "8B"
        top_val = m.group("top")        # e.g. "20"
        start_val = m.group("start")    # e.g. "11"
        end_val = m.group("end")        # e.g. "32"

        # Check if size matches, top value equals our top, and (start, end) is in our pairs.
        if parsed_size != size:
            continue
        if not top_val or int(top_val) != top:
            continue
        if not start_val or not end_val:
            continue
        if (start_val, end_val) not in start_end_pairs:
            continue
        if (start_val, end_val) == (1,33):
            continue

        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        file_path = os.path.join(layer_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        none_key = f"none {task_name}"
        char_key = task_name

        none_stats = extract_stats(data, none_key)
        char_stats = extract_stats(data, char_key)
        acc_none, e_none = compute_acc_e(none_stats)
        acc_char, e_char = compute_acc_e(char_stats)

        # Store in results_dict using the (start_val, end_val) key.
        results_dict[domain]["none"][(start_val, end_val)][task_name].append((acc_none, e_none))
        results_dict[domain]["expert"][(start_val, end_val)][task_name].append((acc_char, e_char))


# ============ 8) Aggregation function ============
def aggregate_condition(data_dict):
    """
    data_dict: { task_name -> list of (acc, e_ratio) }
    Returns (avg_correct, avg_e, avg_incorrect).
    """
    all_acc = []
    all_e = []
    for _, values in data_dict.items():
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
csv_rows = []

# First, save all results for the "None" group
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
    # For each start_end pair, use the fixed top value (top_list[0], which is "20")
    for (s_val, e_val) in start_end_pairs:
        condition_name = f"top_{top}_{s_val}_{e_val}"
        none_agg = aggregate_condition(results_dict[domain]["none"][(s_val, e_val)])
        csv_rows.append({
            "Domain": domain,
            "Group": "None",
            "Condition": condition_name,
            "Avg_Correct": none_agg[0],
            "Avg_E": none_agg[1],
            "Avg_Incorrect": none_agg[2]
        })

# Next, save all results for the "Expert" group
for domain in domain_keys:
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
    # For each start_end pair, again use the fixed top value "20"
    for (s_val, e_val) in start_end_pairs:
        condition_name = f"top_{top}_{s_val}_{e_val}"
        expert_agg = aggregate_condition(results_dict[domain]["expert"][(s_val, e_val)])
        csv_rows.append({
            "Domain": domain,
            "Group": "Expert",
            "Condition": condition_name,
            "Avg_Correct": expert_agg[0],
            "Avg_E": expert_agg[1],
            "Avg_Incorrect": expert_agg[2]
        })

csv_output_path = os.path.join(output_dir, f"aggregated_results_{model}_{size}_multiple_starts.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Domain", "Group", "Condition", "Avg_Correct", "Avg_E", "Avg_Incorrect"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print(f"Aggregated results saved to CSV: {csv_output_path}")


# ================= PLOT LAYER RESULTS =================
plt.style.use('seaborn-v0_8-whitegrid')  # Paper style

# First, read out the baseline (Original) and construct the differences dictionary
baseline = {}
for row in csv_rows:
    domain = row["Domain"]
    group = row["Group"]  # "None" or "Expert"
    condition = row["Condition"]  # "Original" or "top_20_x_y"

    if condition == "Original":
        # Store the original (acc, e) to compute the difference later
        if domain not in baseline:
            baseline[domain] = {}
        baseline[domain][group] = (float(row["Avg_Correct"]), float(row["Avg_E"]))

# differences[domain][group][(start_val, end_val)] = (diff_acc, diff_e)
differences = {}
for row in csv_rows:
    domain = row["Domain"]
    group = row["Group"]
    condition = row["Condition"]
    
    # Skip "Original"
    if condition == "Original":
        continue
    
    # The remaining conditions should be strings like "top_20_1_11"
    # Split to get (start_val, end_val)
    parts = condition.split("_")  # ["top", "20", "startVal", "endVal"]
    if len(parts) != 4:
        continue  # Skip if not in expected format

    # Parse the range
    start_val = parts[2]
    end_val   = parts[3]

    acc = float(row["Avg_Correct"])
    e_  = float(row["Avg_E"])
    
    # Compute the difference from baseline
    base_acc, base_e = baseline[domain][group]
    diff_acc = acc - base_acc
    diff_e   = e_  - base_e
    
    if domain not in differences:
        differences[domain] = {}
    if group not in differences[domain]:
        differences[domain][group] = {}
    differences[domain][group][(start_val, end_val)] = (diff_acc, diff_e)

# Layer range order to plot (customizable order)
layer_pairs_order = [("1", "11"), ("11", "21"), ("21", "32"), ("11", "32"), ("1", "32"), ("1", "33")]

# Calculate overall averages: for each group and (start_val, end_val) range, compute the average across all domains
overall = {"None": {}, "Expert": {}}
for group in ["None", "Expert"]:
    for pair in layer_pairs_order:
        acc_list = []
        e_list   = []
        for dom in differences:
            if group in differences[dom] and pair in differences[dom][group]:
                da, de = differences[dom][group][pair]
                acc_list.append(da)
                e_list.append(de)
        if acc_list:
            mean_acc = np.mean(acc_list)
            mean_e   = np.mean(e_list)
        else:
            # If there is no data for a certain range, set it to 0
            mean_acc, mean_e = 0.0, 0.0
        overall[group][pair] = (mean_acc, mean_e)

# Start plotting
for group in ["None", "Expert"]:
    plt.figure(figsize=(10, 6))
    
    # Plot curves for each domain
    # The x-axis uses integer indices [0, 1, 2, 3, 4], then labels it as "1-11", "11-21", ...
    x_indices = list(range(len(layer_pairs_order)))
    
    for dom in differences:
        if group not in differences[dom]:
            continue
        
        # Extract (diff_acc, diff_e) for the domain across the 5 ranges
        domain_diff_acc = []
        domain_diff_e   = []
        for pair in layer_pairs_order:
            if pair in differences[dom][group]:
                da, de = differences[dom][group][pair]
            else:
                da, de = 0.0, 0.0  # If no data, set to 0
            domain_diff_acc.append(da)
            domain_diff_e.append(de)
        
        # Plot two lines: acc and e
        # Semi-transparent lines show the trend for each domain
        plt.plot(x_indices, domain_diff_acc, marker='o', linestyle='--', alpha=0.4, label=f"{dom} Acc")
        plt.plot(x_indices, domain_diff_e, marker='s', linestyle=':', alpha=0.4, label=f"{dom} E")
    
    # Plot overall mean curve (with thicker lines)
    overall_acc = [overall[group][p][0] for p in layer_pairs_order]
    overall_e   = [overall[group][p][1] for p in layer_pairs_order]
    plt.plot(x_indices, overall_acc, marker='o', color='black', linewidth=2.5, label="Overall Acc Mean")
    plt.plot(x_indices, overall_e, marker='s', color='red',   linewidth=2.5, label="Overall E Mean")
    
    # Set axis labels and title
    tick_labels = [f"[{s},{int(e)-1}]" for (s, e) in layer_pairs_order]
    plt.xticks(x_indices, tick_labels)
    plt.xlabel("Layer Range", fontsize=12, fontweight="bold")
    plt.ylabel("Difference (Modified - Original)", fontsize=12, fontweight="bold")
    plt.title(f"{group}: Layer-based Differences (llama3 8B, top {top})", fontsize=14, fontweight="bold")
    plt.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.show()