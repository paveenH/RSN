#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare 'none_char (modified)' with 'char (original)' across different tasks and model sizes.

Author: paveenhuang
"""

import os
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# =======================
# Configuration
# =======================
model = "phi_v4"
size = "3.8B"
top = 15
alpha = 5

# original data (char)
original_name = "answer_honest_revised"
original_dir = os.path.join(os.getcwd(), f"{model}/{original_name}")

# modified data (none_char)
modified_name = "answer_modified_revised"
modified_name = f"answer_modified_alpha{alpha}_revised"
modified_dir = os.path.join(os.getcwd(), f"{model}/{modified_name}")

# output directory for plots
output_dir = os.path.join(os.getcwd(), f"{model}/counts")
os.makedirs(output_dir, exist_ok=True)

# Regex to match filenames like "abstract_algebra_3B_answers.json" or with optional "_\d"
pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers(_\d+)?\.json$")

original_data = defaultdict(lambda: defaultdict(lambda: {
    'char_acc': 0.0,
    'char_E': 0,
    'total': 0
}))

modified_data = defaultdict(lambda: defaultdict(lambda: {
    'none_char_acc': 0.0,
    'none_char_E': 0,
    'total': 0
}))

# =======================
# 1) Read original char
# =======================
for file in os.listdir(original_dir):
    if file.endswith(f"_{size}_answers.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            size = match.group("size")
            file_path = os.path.join(original_dir, file)

            # JSON
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[original] JSONDecodeError in {file}: {e}")
                continue

            task_name = task.replace('_', ' ')
            accuracy_dict = data.get("accuracy", {})

            for character_type, metrics in accuracy_dict.items():
                if not character_type.startswith("none"):  
                    char_acc = metrics.get("accuracy_percentage", 0.0)
                    char_E = metrics.get("E_count", 0)
                    total = metrics.get("total", 0)

                    original_data[size][task_name]['char_acc'] = char_acc
                    original_data[size][task_name]['char_E'] = char_E
                    original_data[size][task_name]['total'] = total

# =======================
# 2) Read modified none char
# =======================
for file in os.listdir(modified_dir):
    if file.endswith(f"_answers_{top}.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            size = match.group("size")
            file_path = os.path.join(modified_dir, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[modified] JSONDecodeError in {file}: {e}")
                continue

            task_name = task.replace('_', ' ')
            accuracy_dict = data.get("accuracy", {})

            for character_type, metrics in accuracy_dict.items():
                if character_type.startswith("none "):
                    none_char_acc = metrics.get("accuracy_percentage", 0.0)
                    none_char_E = metrics.get("E_count", 0)
                    total = metrics.get("total", 0)

                    modified_data[size][task_name]['none_char_acc'] = none_char_acc
                    modified_data[size][task_name]['none_char_E'] = none_char_E
                    modified_data[size][task_name]['total'] = total

# =======================
# 3) Merge data & visualize
# =======================
for size in sorted(set(list(original_data.keys()) + list(modified_data.keys()))):
    # Build a list to collect all (task, none_char_acc_mdf, char_acc_orig, none_char_E_mdf, char_E_orig, total)
    plot_data = []

    # Collect all tasks
    tasks_orig = set(original_data[size].keys())
    tasks_mdf  = set(modified_data[size].keys())
    all_tasks = sorted(tasks_orig.union(tasks_mdf))

    for task_name in all_tasks:
        none_char_acc_mdf = modified_data[size][task_name]['none_char_acc']
        none_char_E_mdf   = modified_data[size][task_name]['none_char_E']
        total_mdf         = modified_data[size][task_name]['total']

        char_acc_orig = original_data[size][task_name]['char_acc']
        char_E_orig   = original_data[size][task_name]['char_E']
        total_orig    = original_data[size][task_name]['total']

        if total_orig != total_mdf:
            print (f"inconsistency between total_orig {total_orig} and total_mdf {total_mdf} {task_name}")

        plot_data.append({
            'task': task_name,
            'none_char_acc_mdf': none_char_acc_mdf,
            'char_acc_orig': char_acc_orig,
            'none_char_E_pct_mdf': (none_char_E_mdf / total_mdf * 100) if total_mdf > 0 else 0,
            'char_E_pct_orig': (char_E_orig / total_orig * 100) if total_orig > 0 else 0,
            'total': total_mdf
        })

    if not plot_data:
        continue

    df = pd.DataFrame(plot_data).sort_values(by='task').reset_index(drop=True)
    
    # Calculate average accuracy
    avg_non_char_acc = df["none_char_acc_mdf"].mean()
    avg_char_acc = df["char_acc_orig"].mean()

    # Calculate average E count percentage for non-char and char
    avg_none_char_E_pct = df["none_char_E_pct_mdf"].mean()
    avg_char_E_pct = df["char_E_pct_orig"].mean()
    

    print(f"Size: {size}")
    print(f"  Average none-char(modified) Accuracy: {avg_non_char_acc:.2f}%")
    print(f"  Average char(original) Accuracy: {avg_char_acc:.2f}%")
    print(f"  Average none-char(modified) E (%): {avg_none_char_E_pct:.2f}%")
    print(f"  Average char(original) E (%): {avg_char_E_pct:.2f}%")

    # Plot begin 
    index = range(len(df))
    bar_width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # up subplot：Accuracy %
    h1 = ax1.bar(index,
                 df['none_char_acc_mdf'],
                 bar_width,
                 label='none-char(mdf) Accuracy',
                 color='cornflowerblue')
    h2 = ax1.bar([i + bar_width for i in index],
                 df['char_acc_orig'],
                 bar_width,
                 label='char(original) Accuracy',
                 color='salmon')
    ax1.set_ylabel("Acc(%)")
    ax1.set_title(f"Compare none-char(mdf) vs. char(original)\nModel: {model}, Size: {size}, Top: {top}")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # bottom subplot：E Count %
    h3 = ax2.bar(index,
                 df['none_char_E_pct_mdf'],
                 bar_width,
                 label='none-char(mdf) E Counts (%)',
                 color='lightblue')
    h4 = ax2.bar([i + bar_width for i in index],
                 df['char_E_pct_orig'],
                 bar_width,
                 label='char(original) E Counts (%)',
                 color='pink')
    ax2.set_ylabel("E(%)")
    ax2.invert_yaxis()

    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # X
    ax2.set_xticks([i + bar_width / 2 for i in index])
    ax2.set_xticklabels(df['task'], rotation=90)

    # legend
    fig.legend(
        [h1, h2, h3, h4],
        [
            'none-char(mdf) Acc (%)',
            'char(original) Acc (%)',
            'none-char(mdf) E (%)',
            'char(original) E (%)'
        ],
        loc='upper center',
        bbox_to_anchor=(1.1, 0.98),
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, f"compare_none_char_mdf_vs_char_orig_{size}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[Saved Figure] {plot_path}")

print("Done.")