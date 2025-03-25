#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:04:13 2025

@author: paveen
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
from collections import defaultdict

# ---------------------------
# 1) Define the four major areas and their task lists
# ---------------------------
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

# ---------------------------
# 2) Basic configuration: directories and regular expressions
# ---------------------------
model = "llama3_v3"
size = "8B"
top_org = "20"
top = "20"

answer_layer = "answer_modified_layer_revised"   # layer
answer_original = "answer_honest_revised"           # original
answer_mdf = "answer_modified_revised"              # modified all layers

base_path = os.getcwd()
original_dir = os.path.join(base_path, model, answer_original)
layer_dir = os.path.join(base_path, model, answer_layer)
mdf_dir = os.path.join(base_path, model, answer_mdf)

output_dir = os.path.join(base_path, model, "counts")
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(
    r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers"
    r"(?:_(?P<top>\d+))?"
    r"(?:_(?P<start>\d+)_(?P<end>\d+))?"
    r"\.json$"
)

# ---------------------------
# 3) Data structure
# We store (acc, e_ratio) in each domain and each scenario according to the task
# The scenarios include:
#   original_none, original_char, mdf_none, layer_none: dict of segment -> defaultdict(list)
# ---------------------------
domain_keys = list(categories_map.keys())
results_dict = {}
for domain in domain_keys:
    results_dict[domain] = {
        "original_none": defaultdict(list),   # key: task_name -> list of (acc, e_ratio)
        "original_char": defaultdict(list),
        "mdf_none": defaultdict(list),
        "layer_none": {
            "0_4": defaultdict(list),
            "4_8": defaultdict(list),
            "8_12": defaultdict(list),
            "12_16": defaultdict(list),
            "16_20": defaultdict(list),
            "20_24": defaultdict(list),
            "24_28": defaultdict(list),
            "28_32": defaultdict(list),
        }
    }

def get_domain(task_name):
    for dom, tasks in categories_map.items():
        if task_name in tasks:
            return dom
    return "Other"

def compute_acc_e(stats):
    """
    stats: {"correct", "total", "E_count"}
    Returns (acc, e_ratio)
    """
    total = stats.get("total", 0)
    if total == 0:
        return 0.0, 0.0
    acc = stats.get("correct", 0) / total
    e_ratio = stats.get("E_count", 0) / total
    return acc, e_ratio

def extract_stats(data, key):
    """
    Extract the data corresponding to the key from the accuracy part of the JSON and return the stats dictionary
    """
    stats = data.get("accuracy", {}).get(key, {})
    return {
        "correct": stats.get("correct", 0),
        "total": stats.get("total", 0),
        "E_count": stats.get("E_count", 0)
    }

# ---------------------------
# 4) Read Original: store per task results for none and char
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

# ---------------------------
# 5) Read Layer: store per task results (none only) for each segment
# ---------------------------
if os.path.exists(layer_dir):
    for file in os.listdir(layer_dir):
        if f"{size}_answers_{top}" not in file:
            continue
        m = pattern.match(file)
        if not m:
            continue
        task_raw = m.group("task")
        task_name = task_raw.replace('_', ' ')
        domain = get_domain(task_name)
        start = m.group("start")
        end = m.group("end")
        if (start is None) or (end is None):
            continue
        seg_key = f"{start}_{end}"  # e.g. "0_4"
        file_path = os.path.join(layer_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        none_key = f"none {task_name}"
        none_stats = extract_stats(data, none_key)
        acc_none, e_none = compute_acc_e(none_stats)
        if seg_key in results_dict[domain]["layer_none"]:
            results_dict[domain]["layer_none"][seg_key][task_name].append((acc_none, e_none))

# ---------------------------
# 6) Read MDF: store per task results (none only)
# ---------------------------
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
        none_stats = extract_stats(data, none_key)
        acc_none, e_none = compute_acc_e(none_stats)
        results_dict[domain]["mdf_none"][task_name].append((acc_none, e_none))

# ---------------------------
# 7) Helper: average results over tasks
# ---------------------------
def average_scenario(scenario_dict):
    task_avgs = []
    for task, values in scenario_dict.items():
        if len(values) == 0:
            continue
        accs = [v[0] for v in values]
        es = [v[1] for v in values]
        task_avg_acc = np.mean(accs)
        task_avg_e = np.mean(es)
        task_avgs.append((task_avg_acc, task_avg_e))
    if len(task_avgs) == 0:
        return (0.0, 0.0)
    overall_acc = np.mean([v[0] for v in task_avgs])
    overall_e = np.mean([v[1] for v in task_avgs])
    return (overall_acc, overall_e)

# ---------------------------
# 8) Plot
# X-axis: 11 points:
#   1) Original (none)
#   2-9) 8 segments (none): 0-4, 4-8, 8-12, 12-16, 16-20, 20-24, 24-28, 28-32
#   10) MDF (none)
#   11) Original (char) -- for contrast
# Left Y-axis: Accuracy, Right Y-axis: E Ratio
# ---------------------------
ordered_segments = [
    ("original_none", "Org(N)"),
    ("0_4", "0-4(N)"),
    ("4_8", "4-8(N)"),
    ("8_12", "8-12(N)"),
    ("12_16", "12-16(N)"),
    ("16_20", "16-20(N)"),
    ("20_24", "20-24(N)"),
    ("24_28", "24-28(N)"),
    ("28_32", "28-32(N)"),
    ("mdf_none", "MDF(N)"),
    ("original_char", "Org(Expert)")
]

def plot_domain_results(domain_name, data_dict, save_dir):
    x_labels = []
    acc_values = []
    e_values = []
    for key, label in ordered_segments:
        if key == "original_none":
            avg = average_scenario(data_dict["original_none"])
        elif key == "original_char":
            avg = average_scenario(data_dict["original_char"])
        elif key == "mdf_none":
            avg = average_scenario(data_dict["mdf_none"])
        else:
            # For layer segments:
            avg = average_scenario(data_dict["layer_none"].get(key, {}))
        x_labels.append(label)
        acc_values.append(avg[0])
        e_values.append(avg[1])
    
    x = np.arange(len(ordered_segments))  # 0 .. 10
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Updated colors for publication
    bar_color = "#4878D0"    # Medium blue - professional and visible
    line_color = "#EE854A"   # Orange - contrasts well with blue
    
    # Modify style for better visibility
    ax1.bar(x, acc_values, width=0.6, label="Accuracy", alpha=0.85, 
            color=bar_color, edgecolor='black', linewidth=0.5)
    
    # Enhanced line plot
    ax2.plot(x, e_values, marker='o', color=line_color, label="E Ratio",
             linewidth=2, markersize=8, markeredgecolor='black', 
             markeredgewidth=0.5)
    
    # Style improvements
    ax1.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax2.set_ylabel("E Ratio", fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    
    # X-axis styling
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha='right')
    
    # Title styling
    ax1.set_title(f"{domain_name} - Accuracy & E Ratio (Llama3-8B, top {top} neurons) ",  
                  fontsize=14, fontweight='bold', pad=15)
    
    # Legend improvements
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper right', fontsize=10, frameon=True, 
              edgecolor='black')
    
    # Grid styling
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--', color='gray')
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high resolution
    save_path = os.path.join(save_dir, f"{domain_name.replace(' ', '_')}_11points_none_char.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    # Also save as PNG for quick viewing
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    print(f"Saved figures as:\n{save_path}\n{png_path}")

# ---------------------------
# 9) Plot for each domain
# ---------------------------
for domain, data in results_dict.items():
    plot_domain_results(domain, data, output_dir)

print("Done! All domain figures have been saved.")

# ---------------------------
# 10) Save Data
# ---------------------------
def save_all_domain_results_to_csv(results_dict, save_dir):

    rows = []
    for domain, data_dict in results_dict.items():
        for key, label in ordered_segments:
            if key == "original_none":
                avg = average_scenario(data_dict["original_none"])
            elif key == "original_char":
                avg = average_scenario(data_dict["original_char"])
            elif key == "mdf_none":
                avg = average_scenario(data_dict["mdf_none"])
            else:
                avg = average_scenario(data_dict["layer_none"].get(key, {}))
            rows.append({
                "Domain": domain,
                "Segment": label,
                "Accuracy": avg[0],
                "E_Ratio": avg[1]
            })
    csv_path = os.path.join(save_dir, "all_domains_results.csv")
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Domain", "Segment", "Accuracy", "E_Ratio"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved combined CSV data: {csv_path}")

save_all_domain_results_to_csv(results_dict, output_dir)