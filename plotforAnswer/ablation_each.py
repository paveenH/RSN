#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:57:44 2025

@author: paveenhuang
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
from collections import defaultdict
from matplotlib.ticker import MaxNLocator



# ---------------------------
# 1) Define the 5 tasks
# ---------------------------
tasks = [
    "abstract algebra",
    "anatomy",
    "astronomy",
    "global facts",
    "econometrics",
    "jurisprudence"
]

# ---------------------------
# 2) Basic configuration: directories and regular expressions
# ---------------------------
model = "llama3_v3"
size = "8B"
top = "20"  # Choose number of modified neurosn, 20 or 640

answer_ablation = "answer_ablation_revised"   # index modifications 
# answer_ablation = "answer_ablation_reverse_revised"   # index modifications 

answer_original = "answer_honest_revised"           # original
answer_mdf = "answer_modified_revised"              # modified all layers

base_path = os.getcwd()
original_dir = os.path.join(base_path, model, answer_original)
ablation_dir = os.path.join(base_path, model, answer_ablation)
mdf_dir = os.path.join(base_path, model, answer_mdf)

output_dir = os.path.join(base_path, model, "counts")
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(
    r"^(?P<task>.+)_" 
    r"(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers"
    r"(?:_(?P<top>\d+))?"
    r"(?:_(?P<start>\d+)_(?P<end>\d+))?"
    r"(?:_(?P<index>\d+))?"
    r"\.json$"
)

# ---------------------------
# 3) Data structure
# For each task, save:
#   "original": defaultdict(list)
#   "mdf": defaultdict(list)  
#   "layer": defaultdict(list)  
# ---------------------------
results_dict = {}
for task in tasks:
    results_dict[task] = {
         "original": defaultdict(list),
         "mdf": defaultdict(list),
         "ablation": defaultdict(list)
    }

def compute_acc_e(stats):
    total = stats.get("total", 0)
    if total == 0:
        return 0.0, 0.0
    acc = stats.get("correct", 0) / total
    e_ratio = stats.get("E_count", 0) / total
    return acc, e_ratio

def extract_stats(data, key):
    stats = data.get("accuracy", {}).get(key, {})
    return {
         "correct": stats.get("correct", 0),
         "total": stats.get("total", 0),
         "E_count": stats.get("E_count", 0)
    }

# ---------------------------
# 4) Read Original: store results for each task (none)
# ---------------------------
if os.path.exists(original_dir):
    for file in os.listdir(original_dir):
        if f"{size}_answers" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        if task_name not in tasks:
            continue
        file_path = os.path.join(original_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        none_key = f"none {task_name}"
        stats = extract_stats(data, none_key)
        acc, e = compute_acc_e(stats)
        results_dict[task_name]["original"][task_name].append((acc, e))

# ---------------------------
# 5) Read MDF: store results for each task (none)
# ---------------------------
if os.path.exists(mdf_dir):
    for file in os.listdir(mdf_dir):
        if f"{size}_answers_{top}" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        if task_name not in tasks:
            continue
        file_path = os.path.join(mdf_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        none_key = f"none {task_name}"
        stats = extract_stats(data, none_key)
        acc, e = compute_acc_e(stats)
        results_dict[task_name]["mdf"][task_name].append((acc, e))

# ---------------------------
# 6) Read Layer: store single-layer modification results for each task (none)
# ---------------------------
if os.path.exists(ablation_dir):
    for file in os.listdir(ablation_dir):
        if f"{size}_answers_{top}" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        if task_name not in tasks:
            continue
        index = m.group("index")   
        seg_key = f"{index}"  
        file_path = os.path.join(ablation_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        none_key = f"none {task_name}"
        stats = extract_stats(data, none_key)
        acc, e = compute_acc_e(stats)
        results_dict[task_name]["ablation"][seg_key].append((acc, e))

def average_values(values):
    if not values:
        return 0.0, 0.0
    accs = [v[0] for v in values]
    es = [v[1] for v in values]
    return np.mean(accs), np.mean(es)

# ---------------------------
# 7) Plot for each task based on ablation index results
# X-axis: layers (1-indexed, e.g., 1 to N) based on ablation data keys
# We use two line charts: one for Accuracy, one for E Ratio.
# Additionally, add horizontal reference lines for Original (none) and MDF (none).
# ---------------------------
def plot_task_results(task_name, data_dict, save_dir):
    layer_keys = sorted(data_dict["ablation"].keys(), key=lambda x: int(x))
    x = np.arange(1, len(layer_keys)+1)  # 1-indexed layer numbers
    
    ablation_acc = []
    ablation_e = []
    for key in layer_keys:
        avg = average_values(data_dict["ablation"][key])
        ablation_acc.append(avg[0])
        ablation_e.append(avg[1])
    
    # Compute reference values (average over original and MDF)
    orig_avg = average_values(data_dict["original"][task_name])
    mdf_avg = average_values(data_dict["mdf"][task_name])
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Draw the ablation modification effect curves
    ax1.plot(x, ablation_acc, marker='o', color="#4878D0", linewidth=2, markersize=6, label="Ablation (Acc)")
    ax2.plot(x, ablation_e, marker='o', color="#EE854A", linewidth=2, markersize=6, label="Ablation (E Ratio)")
    
    # Add horizontal guides for Original and MDF
    ax1.axhline(orig_avg[0], color="green", linestyle="--", linewidth=2, label="Original (Acc)")
    ax1.axhline(mdf_avg[0], color="red", linestyle="--", linewidth=2, label="MDF (Acc)")
    ax2.axhline(orig_avg[1], color="green", linestyle=":", linewidth=2, label="Original (E Ratio)")
    ax2.axhline(mdf_avg[1], color="red", linestyle=":", linewidth=2, label="MDF (E Ratio)")
    
    ax1.set_xlabel("Ablation Index (Sorted Order)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax2.set_ylabel("E Ratio", fontsize=12, fontweight='bold')
    ax1.set_title(f"{task_name.title()} - Ablation Results (top {top} neurons)", fontsize=14, fontweight='bold')
    
    ax1.set_xlim([1, len(layer_keys)])
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10, frameon=True, edgecolor='black')
    
    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{task_name.replace(' ', '_')}_ablation_top{top}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved figure for {task_name}: {save_path}")

for task in tasks:
    plot_task_results(task, results_dict[task], output_dir)

# ---------------------------
# 8) Save all tasks ablation results to a combined CSV
# ---------------------------
def save_all_tasks_results_to_csv(results_dict, save_dir):
    rows = []
    for task, data_dict in results_dict.items():
        for key in sorted(data_dict["ablation"].keys(), key=lambda x: int(x)):
            avg = average_values(data_dict["ablation"][key])
            ablation_index = int(key) 
            rows.append({
                "Task": task,
                "Ablation_Index": ablation_index,
                "Accuracy": avg[0],
                "E_Ratio": avg[1]
            })
    csv_path = os.path.join(save_dir, "all_tasks_ablation_results.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Task", "Ablation_Index", "Accuracy", "E_Ratio"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved combined CSV data: {csv_path}")

save_all_tasks_results_to_csv(results_dict, output_dir)