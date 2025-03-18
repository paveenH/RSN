#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:00:50 2025

@author: paveenhuang
"""


import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import re

# ---------------------------
# 1) Define categories and task mappings
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
# 2) Define data directories
# ---------------------------
top = 20
size = "8B"
model = "llama3_v3"
data_dir = os.path.join(os.getcwd(), f"{model}/answer_honest_revised")
# modified_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_revised")
index = "1731"
modified_dir = os.path.join(os.getcwd(), f"{model}/answer_lesion_revised")

# ---------------------------
# 3) Define data structures for storing read results
# ---------------------------
# results: Original results (none_char / char)
results = defaultdict(lambda: {"none_character": {}, "character": {}})
results_mdf = defaultdict(lambda: {"none_character_mdf": {}, "character_mdf": {}})

# pattern = re.compile(r"^(?P<task>.+)_(?P<size>1B|3B|8B)_answers\.json$")
pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3.8B|7B|8B)_answers(_\d+)?\.json$")


# ---------------------------
# 4) Read JSON files from the original directory and save to results
# ---------------------------
for file in os.listdir(data_dir):
    if file.endswith(f"{size}_answers.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            size = match.group("size")
            file_path = os.path.join(data_dir, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            task_name = task.replace('_', ' ')
            accuracy = data.get("accuracy", {})
            none_char_key = f"none {task_name}"
            char_key = task_name

            none_char_data = accuracy.get(none_char_key, {})
            char_data = accuracy.get(char_key, {})

            results[task_name]["none_character"][size] = {
                "correct": none_char_data.get("correct", 0),
                "total": none_char_data.get("total", 0),
                "E_count": none_char_data.get("E_count", 0),
            }
            results[task_name]["character"][size] = {
                "correct": char_data.get("correct", 0),
                "total": char_data.get("total", 0),
                "E_count": char_data.get("E_count", 0),
            }
        else:
            print(f"Filename does not match pattern and will be skipped: {file}")

# ---------------------------
# 5) Read JSON files from the modified directory and save to results_mdf
#    (Note: The only difference here is in the storage keys: none_character_mdf / character_mdf)
# ---------------------------
for file in os.listdir(modified_dir):
    if file.endswith(f"{size}_answers_{index}.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            size = match.group("size")
            file_path = os.path.join(modified_dir, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            task_name = task.replace('_', ' ')
            accuracy = data.get("accuracy", {})
            none_char_key = f"none {task_name}"
            char_key = task_name

            none_char_data = accuracy.get(none_char_key, {})
            char_data = accuracy.get(char_key, {})

            results_mdf[task_name]["none_character_mdf"][size] = {
                "correct": none_char_data.get("correct", 0),
                "total": none_char_data.get("total", 0),
                "E_count": none_char_data.get("E_count", 0),
            }
            results_mdf[task_name]["character_mdf"][size] = {
                "correct": char_data.get("correct", 0),
                "total": char_data.get("total", 0),
                "E_count": char_data.get("E_count", 0),
            }
        else:
            print(f"Filename does not match pattern and will be skipped: {file}")

# ---------------------------
# 6) Create a new data structure, category_data, to summarize all four results
# ---------------------------
category_data = {
    cat: {
        "none_character": defaultdict(lambda: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []}),
        "none_character_mdf": defaultdict(lambda: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []}),
        "character": defaultdict(lambda: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []}),
        "character_mdf": defaultdict(lambda: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []}),
    }
    for cat in categories_map
}

# Helper function: Find the category for a given task_name
def find_category(task_name):
    for c, task_list in categories_map.items():
        if task_name in task_list:
            return c
    return None

# ---------------------------
# 7) Summarize original results (none_character / character)
# ---------------------------
for task_name, stat_dict in results.items():
    category = find_category(task_name)
    if category is None:
        print(f"[Warning] '{task_name}' not found in any category.")
        continue
    
    # none_character
    for size, info in stat_dict["none_character"].items():
        total_ = info["total"]
        correct_ = info["correct"]
        e_ = info["E_count"]
        if total_ > 0:
            correct_ratio = correct_ / total_
            e_ratio = e_ / total_
            incorrect_ratio = 1.0 - correct_ratio - e_ratio
            
            category_data[category]["none_character"][size]["correct_ratios"].append(correct_ratio)
            category_data[category]["none_character"][size]["e_ratios"].append(e_ratio)
            category_data[category]["none_character"][size]["incorrect_ratios"].append(incorrect_ratio)
    
    # character
    for size, info in stat_dict["character"].items():
        total_ = info["total"]
        correct_ = info["correct"]
        e_ = info["E_count"]
        if total_ > 0:
            correct_ratio = correct_ / total_
            e_ratio = e_ / total_
            incorrect_ratio = 1.0 - correct_ratio - e_ratio
            
            category_data[category]["character"][size]["correct_ratios"].append(correct_ratio)
            category_data[category]["character"][size]["e_ratios"].append(e_ratio)
            category_data[category]["character"][size]["incorrect_ratios"].append(incorrect_ratio)

# ---------------------------
# 8) Summarize modified results (none_character_mdf / character_mdf)
# ---------------------------
for task_name, stat_dict in results_mdf.items():
    category = find_category(task_name)
    if category is None:
        print(f"[Warning] '{task_name}' not found in any category. (modified set)")
        continue
    
    # none_character_mdf
    for size, info in stat_dict["none_character_mdf"].items():
        total_ = info["total"]
        correct_ = info["correct"]
        e_ = info["E_count"]
        if total_ > 0:
            correct_ratio = correct_ / total_
            e_ratio = e_ / total_
            incorrect_ratio = 1.0 - correct_ratio - e_ratio
            
            category_data[category]["none_character_mdf"][size]["correct_ratios"].append(correct_ratio)
            category_data[category]["none_character_mdf"][size]["e_ratios"].append(e_ratio)
            category_data[category]["none_character_mdf"][size]["incorrect_ratios"].append(incorrect_ratio)
    
    # character_mdf
    for size, info in stat_dict["character_mdf"].items():
        total_ = info["total"]
        correct_ = info["correct"]
        e_ = info["E_count"]
        if total_ > 0:
            correct_ratio = correct_ / total_
            e_ratio = e_ / total_
            incorrect_ratio = 1.0 - correct_ratio - e_ratio
            
            category_data[category]["character_mdf"][size]["correct_ratios"].append(correct_ratio)
            category_data[category]["character_mdf"][size]["e_ratios"].append(e_ratio)
            category_data[category]["character_mdf"][size]["incorrect_ratios"].append(incorrect_ratio)


# ---------------------------
# 9) Plot results for all categories in a single figure
# ---------------------------

# Set global font style for a paper-friendly appearance
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.5,  # Thicker axis lines for better readability
    "figure.figsize": (16, 8),  # Larger figure
    "axes.grid": False,  # Remove unnecessary grid
})

# Define a professional color palette
colors = {
    'none_character': {
        'correct': '#FFD48E',    # lighter orange
        'e': '#7DA1C4',          # lighter blue
        'incorrect': '#808080'   # lighter gray
    },
    'none_character_mdf': {
        'correct': '#FFA07A',    # light salmon
        'e': '#87CEFA',          # light sky blue
        'incorrect': '#A9A9A9'   # dark gray
    },
    'character': {
        'correct': '#FF8C00',    # deeper orange
        'e': '#2E5A88',          # deeper blue
        'incorrect': '#404040'   # deeper gray
    },
    'character_mdf': {
        'correct': '#FF4500',    # orange red
        'e': '#1E90FF',          # dodger blue
        'incorrect': '#2F4F4F'   # dark slate gray
    }
}

# Define categories and groups
categories = list(categories_map.keys())
group_names = ["none_character", "none_character_mdf", "character", "character_mdf"]
num_groups = len(group_names)
bar_width = 0.18

# X-axis positions for each category
x = np.arange(len(categories))

# Compute spacing for each group within a category
offsets = np.linspace(- (num_groups - 1) * (bar_width / 2), 
                      (num_groups - 1) * (bar_width / 2), 
                      num_groups)

# Create a new figure for plotting
fig, ax = plt.subplots(figsize=(16, 8))

# Iterate over groups to plot stacked bars
for i, group in enumerate(group_names):
    bar_positions = x + offsets[i]  # Adjust positions within categories

    avg_correct_ratios, avg_e_ratios, avg_incorrect_ratios = [], [], []

    # Collect aggregated data for each category
    for category in categories:
        metrics = category_data[category][group].get(size, {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []})

        avg_correct = np.mean(metrics["correct_ratios"]) if metrics["correct_ratios"] else 0.0
        avg_e = np.mean(metrics["e_ratios"]) if metrics["e_ratios"] else 0.0
        avg_incorrect = np.mean(metrics["incorrect_ratios"]) if metrics["incorrect_ratios"] else 0.0

        avg_correct_ratios.append(avg_correct)
        avg_e_ratios.append(avg_e)
        avg_incorrect_ratios.append(avg_incorrect)

    color_set = colors[group]  # Assign color set

    # Plot stacked bars
    ax.bar(bar_positions, avg_correct_ratios, bar_width, 
           color=color_set['correct'], edgecolor="black", label=group if i == 0 else "")
    ax.bar(bar_positions, avg_e_ratios, bar_width, 
           bottom=avg_correct_ratios, color=color_set['e'], edgecolor="black")
    ax.bar(bar_positions, avg_incorrect_ratios, bar_width, 
           bottom=np.array(avg_correct_ratios) + np.array(avg_e_ratios), 
           color=color_set['incorrect'], edgecolor="black")

# X-axis formatting
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=14)
ax.set_ylabel("Proportion of Answers", fontsize=16)
ax.set_title(f"Performance Comparison for {model[:-3]} {size} (Top {top} Neurons)", fontsize=18, fontweight="bold")

# Remove unnecessary spines
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Adjusted legend elements with 'facecolor' instead of 'color'
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['none_character']['correct'], edgecolor='black', label='None-Expert (O)'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['none_character_mdf']['correct'], edgecolor='black', label='None-Expert (M)'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['character']['correct'], edgecolor='black', label='Expert (O)'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors['character_mdf']['correct'], edgecolor='black', label='Expert (M)')
]

legend_types = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#FFD48E', edgecolor='black', label='Correct'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#7DA1C4', edgecolor='black', label='E Responses'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#808080', edgecolor='black', label='Incorrect (not E)')
]

# Adjust legend positions
legend1 = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10)
legend2 = ax.legend(handles=legend_types, loc="upper left", bbox_to_anchor=(1.0, 0.75), fontsize=10)
ax.add_artist(legend1)  # Ensure both legends appear

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for the legend

# Save figure
output_dir = os.path.join(os.getcwd(), f"{model}/counts")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"performance_comparison_{size}_top_{top}.png")
plt.savefig(output_path, bbox_inches="tight", dpi=300)
print(f"Saved plot to {output_path}")

plt.show()
plt.close()
import pandas as pd

# ---------------------------
# 10) Save results to CSV
# ---------------------------
# Prepare data storage
csv_data = []
headers = ["Group"]
for category in categories:
    headers.extend([f"{category}_Mean_Acc", f"{category}_Mean_E"])

# Iterate through all groups and compute average accuracy & E rate
for group in group_names:
    row = [group]  # First column is the Group name
    for category in categories:
        metrics = category_data[category][group].get(size, {"correct_ratios": [], "e_ratios": []})

        # Compute mean accuracy and E rate
        mean_acc = np.mean(metrics["correct_ratios"]) * 100 if metrics["correct_ratios"] else 0.0
        mean_e = np.mean(metrics["e_ratios"]) * 100 if metrics["e_ratios"] else 0.0

        row.extend([f"{mean_acc:.2f}%", f"{mean_e:.2f}%"])  # Format as percentage

    csv_data.append(row)

# Convert to DataFrame and save
df = pd.DataFrame(csv_data, columns=headers)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
csv_output_path = os.path.join(output_dir, f"performance_comparison_{size}_index_{index}.csv")

df.to_csv(csv_output_path, index=False)
print(f"Saved CSV file to {csv_output_path}")