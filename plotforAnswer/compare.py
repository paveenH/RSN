#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:09:42 2024

Author: paveenhuang
"""

import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import re

# --------------------------------------------------
# 1) Define categories and task mappings (for colors)
# --------------------------------------------------
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

# A helper to find which category a task belongs to
def find_category(task_name):
    for c, lst in categories_map.items():
        if task_name in lst:
            return c
    return None

category_colors = {
    "STEM": "#a6cee3",            # Soft Blue
    "Humanities": "#b2df8a",      # Soft Green
    "Social Sciences": "#fdbf6f", # Soft Orange
    "Other": "#cab2d6"             # Soft Purple
}

# --------------------------------------------------
# 2) Define data directories
# --------------------------------------------------
model = "llama3_v3"
size = "1B"
top = 10
# alpha = 5
data_dir = os.path.join(os.getcwd(), f"{model}/answer_honest_revised")
modified_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_revised")
# modified_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_alpha{alpha}_revised")

out_path =  os.path.join(os.getcwd(), f"{model}/counts")
os.makedirs(out_path, exist_ok=True)


# --------------------------------------------------
# 3) Data structures: store each task's char / char_mdf and non_char / non_char_mdf ratios by size
# --------------------------------------------------
# Per-task data, e.g.:
# per_task_data[task_name]["character"][size] = correct_ratio
# per_task_data[task_name]["character_mdf"][size] = correct_ratio
# per_task_data[task_name]["none_character"][size] = correct_ratio
# per_task_data[task_name]["none_character_mdf"][size] = correct_ratio
per_task_data = defaultdict(lambda: {
    "character": {},
    "character_mdf": {},
    "none_character": {},
    "none_character_mdf": {},
})

# Regex for filenames
pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers(_\d+)?\.json$")

# ---------------------------
# 4) Read original JSON files -> fill per_task_data for "character" and "none_character"
# ---------------------------
for file in os.listdir(data_dir):
    if file.endswith(f"{size}_answers.json"):
        match = pattern.match(file)
        if not match:
            print(f"Filename not matching pattern (skipped): {file}")
            continue
        
        task = match.group("task")
        size_matched = match.group("size")
        file_path = os.path.join(data_dir, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        task_name = task.replace('_', ' ')
        accuracy = data.get("accuracy", {})
        
        # Handle "character"
        char_key = task_name  # for "character"
        char_data = accuracy.get(char_key, {})
        correct_ = char_data.get("correct", 0)
        total_ = char_data.get("total", 0)
        if total_ > 0:
            correct_ratio = correct_ / total_
        else:
            correct_ratio = 0.0
        per_task_data[task_name]["character"][size_matched] = correct_ratio

        # Handle "none_character"
        none_char_key = f"none {task_name}"
        none_char_data = accuracy.get(none_char_key, {})
        correct_none = none_char_data.get("correct", 0)
        total_none = none_char_data.get("total", 0)
        if total_none > 0:
            correct_none_ratio = correct_none / total_none
        else:
            correct_none_ratio = 0.0
        per_task_data[task_name]["none_character"][size_matched] = correct_none_ratio

# ---------------------------
# 5) Read modified JSON files -> fill per_task_data for "character_mdf" and "none_character_mdf"
# ---------------------------
for file in os.listdir(modified_dir):
    if file.endswith(f"_answers_{top}.json"):
        match = pattern.match(file)
        if not match:
            print(f"Filename not matching pattern (skipped): {file}")
            continue
        
        task = match.group("task")
        size_matched = match.group("size")
        file_path = os.path.join(modified_dir, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        task_name = task.replace('_', ' ')
        accuracy = data.get("accuracy", {})
        
        # Handle "character_mdf"
        char_mdf_key = task_name  # for "character_mdf"
        char_mdf_data = accuracy.get(char_mdf_key, {})
        correct_mdf = char_mdf_data.get("correct", 0)
        total_mdf = char_mdf_data.get("total", 0)
        if total_mdf > 0:
            correct_mdf_ratio = correct_mdf / total_mdf
        else:
            correct_mdf_ratio = 0.0
        per_task_data[task_name]["character_mdf"][size_matched] = correct_mdf_ratio

        # Handle "none_character_mdf"
        none_char_mdf_key = f"none {task_name}"
        none_char_mdf_data = accuracy.get(none_char_mdf_key, {})
        correct_none_mdf = none_char_mdf_data.get("correct", 0)
        total_none_mdf = none_char_mdf_data.get("total", 0)
        if total_none_mdf > 0:
            correct_none_mdf_ratio = correct_none_mdf / total_none_mdf
        else:
            correct_none_mdf_ratio = 0.0
        per_task_data[task_name]["none_character_mdf"][size_matched] = correct_none_mdf_ratio

# ---------------------------
# 6) Compute differences for the specified size
# ---------------------------
size_changes = {
    "character": [],
    "none_character": []
}

# Loop over all tasks found in per_task_data
all_tasks = list(per_task_data.keys())
for task in all_tasks:
    # Find category for color usage
    cat = find_category(task)
    if cat is None:
        cat = "Other"  # Assign to "Other" if category not found
    
    # Process only the specified size
    # Character
    original_char_ratio = per_task_data[task]["character"].get(size, 0.0)
    modified_char_ratio = per_task_data[task]["character_mdf"].get(size, 0.0)
    change_char = modified_char_ratio - original_char_ratio
    
    size_changes["character"].append({
        "task": task,
        "category": cat,
        "change": change_char
    })
    
    # Non-Character
    original_none_char_ratio = per_task_data[task]["none_character"].get(size, 0.0)
    modified_none_char_ratio = per_task_data[task]["none_character_mdf"].get(size, 0.0)
    change_none_char = modified_none_char_ratio - original_none_char_ratio
    
    size_changes["none_character"].append({
        "task": task,
        "category": cat,
        "change": change_none_char
    })

# ---------------------------
# 7) Plotting function (one plot per metric: character and non_character)
# ---------------------------
def plot_performance_change(metric, size, changes, category_colors):
    """
    Plots the performance change for a given metric (character or none_character).
    - metric (str): 'character' or 'none_character'
    - size (str): '1B', '3B', etc.
    - changes (list): each item has keys "task", "category", "change"
    - category_colors (dict): map from category to color
    """
    # 1) Sort tasks by change (ascending)
    sorted_changes = sorted(changes, key=lambda x: x["change"])
    
    tasks = [item["task"] for item in sorted_changes]
    change_values = [item["change"] for item in sorted_changes]
    cats = [item["category"] for item in sorted_changes]
    bar_colors = [category_colors.get(c, "#333333") for c in cats]  # fallback color
    
    # 2) Plot
    plt.figure(figsize=(20, 10), dpi=300)
    x_positions = np.arange(len(tasks))
    
    bars = plt.bar(x_positions, change_values, color=bar_colors, edgecolor='black')
    
    # Horizontal line at 0
    plt.axhline(0, color='grey', linewidth=1)
    
    # X-axis ticks
    plt.xticks(x_positions, tasks, rotation=90)
    
    # Labels & title
    metric_title = "Non-Character" if metric == "none_character" else "Character"
    plt.xlabel('Tasks', fontsize=14)
    plt.ylabel('Change in Correct Ratio (modified - original)', fontsize=14)
    plt.title(f'{metric_title} vs {metric_title}_mdf Change for Model Size: {size}', fontsize=16)
    
    # 3) Add legend by category
    from matplotlib.patches import Patch
    unique_categories = sorted(list(set(cats)), key=lambda x: list(categories_map.keys()).index(x) if x in categories_map else -1)
    legend_elements = []
    for ckey in unique_categories:
        ccolor = category_colors.get(ckey, "#333333")
        legend_elements.append(Patch(facecolor=ccolor, edgecolor='black', label=ckey))
    plt.legend(handles=legend_elements, title='Categories', loc='upper right')
    
    # 4) Annotate each bar with % change
    for idx, bar in enumerate(bars):
        h = bar.get_height()
        # Decide text position
        va = 'bottom' if h >= 0 else 'top'
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            h,
            f'{h:.2%}',
            ha='center',
            va=va,
            fontsize=8,
            rotation=90
        )
    
    plt.tight_layout()
    
    # 5) Save & show
    out_name = os.path.join(out_path, f"performance_change_{metric}_{size}.png")
    plt.savefig(out_name, bbox_inches='tight', dpi=300)
    print(f"[Saved] {out_name}")
    plt.show()
    plt.close()

# ---------------------------
# 8) Generate plots for the specified size
# ---------------------------
metrics = ["character", "none_character"]

for metric in metrics:
    if metric in size_changes and size_changes[metric]:
        plot_performance_change(metric, size, size_changes[metric], category_colors)
    else:
        print(f"No data available for metric: {metric} and size: {size}")