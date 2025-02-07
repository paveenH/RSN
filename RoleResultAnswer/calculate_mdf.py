#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:09:42 2024

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
top = 15
size = "3.8B"
model = "phi_v4"
alpha = 5
data_dir = os.path.join(os.getcwd(), f"{model}/answer_honest_revised")
# modified_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_revised")
modified_dir = os.path.join(os.getcwd(), f"{model}/answer_modified_alpha{alpha}_revised")

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
    if file.endswith(f"{size}_answers_{top}.json"):
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

plt.figure(figsize=(20, 12), dpi=300)

# Define an expanded color palette
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
        'correct': '#FF8C00',    # darker orange
        'e': '#2E5A88',          # darker blue
        'incorrect': '#404040'   # darker gray
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
group_spacing = 0.0
category_spacing = 0.5

# Calculate the positions for each category
x = np.arange(len(categories))

# Calculate offsets for each group within a category to place them side by side
offsets = np.linspace(- (num_groups - 1) * (bar_width + group_spacing) / 2,
                      (num_groups - 1) * (bar_width + group_spacing) / 2,
                      num_groups)

# Iterate over each group to plot their bars
for i, group in enumerate(group_names):
    # Calculate the position for each bar within the category
    bar_positions = x + offsets[i]
    
    # Aggregate data for each category
    avg_correct_ratios = []
    avg_e_ratios = []
    avg_incorrect_ratios = []
    
    for category in categories:
        metrics = category_data[category][group].get(size, {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []})
        if metrics["correct_ratios"]:
            avg_correct = np.mean(metrics["correct_ratios"])
            avg_e = np.mean(metrics["e_ratios"])
            avg_incorrect = np.mean(metrics["incorrect_ratios"])
        else:
            avg_correct = 0.0
            avg_e = 0.0
            avg_incorrect = 0.0
        avg_correct_ratios.append(avg_correct)
        avg_e_ratios.append(avg_e)
        avg_incorrect_ratios.append(avg_incorrect)
    
    # Use the specific color set for each group
    color_set = colors[group]
    
    # Plot stacked bars
    bars_correct = plt.bar(
        bar_positions, avg_correct_ratios, bar_width,
        color=color_set['correct'], edgecolor="black", label=group if i == 0 else ""
    )
    bars_e = plt.bar(
        bar_positions, avg_e_ratios, bar_width,
        bottom=avg_correct_ratios, color=color_set['e'], edgecolor="black"
    )
    bars_incorrect = plt.bar(
        bar_positions, avg_incorrect_ratios, bar_width,
        bottom=np.array(avg_correct_ratios) + np.array(avg_e_ratios),
        color=color_set['incorrect'], edgecolor="black"
    )

# Customize the x-axis with category labels
plt.xticks(x, categories, rotation=45, fontsize=16, ha='right')
plt.xlabel("Category", fontsize=14)
plt.ylabel("Proportion of Questions", fontsize=16)
plt.title(f"Performance Comparison for Model {model[:-3]} {size} Top {top}", fontsize=18)

# Create a single legend for all stacked components and groups
handles = [
    plt.Rectangle((0,0),1,1, color=colors['none_character']['correct'], label='none_character'),
    plt.Rectangle((0,0),1,1, color=colors['none_character_mdf']['correct'], label='none_character_mdf'),
    plt.Rectangle((0,0),1,1, color=colors['character']['correct'], label='character'),
    plt.Rectangle((0,0),1,1, color=colors['character_mdf']['correct'], label='character_mdf')
]
plt.legend(handles=handles, title="Groups", loc="upper left", fontsize=10, title_fontsize=12)

# Add a secondary legend for bar components
plt.legend(["Correct", "E Responses", "Incorrect (not E)"], 
           loc="upper left", fontsize=12, title="Response Types")

# Adjust bottom margin to prevent cutting off x-axis labels
plt.subplots_adjust(bottom=0.2, top=0.85)

# Adjust layout and save the plot
plt.tight_layout()
output_dir = os.path.join(os.getcwd(), f"{model}")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"performance_comparison_{size}_top_{top}.png")
plt.savefig(output_path, bbox_inches="tight", dpi=300)
print(f"Saved plot to {output_path}")

plt.show()
plt.close()


# ---------------------------
# 10) save data to csv
# ---------------------------

import pandas as pd

group_names = ["none_character", "none_character_mdf", "character", "character_mdf"]
target_size = size
summary_data = []

for category, groups in category_data.items():
    for group in group_names:
        group_metrics = groups.get(group, {}).get(target_size, {})
        correct_ratios = group_metrics.get("correct_ratios", [])
        e_ratios = group_metrics.get("e_ratios", [])
        incorrect_ratios = group_metrics.get("incorrect_ratios", [])
        
        if correct_ratios and e_ratios and incorrect_ratios:
            mean_correct = np.mean(correct_ratios) * 100  # 转换为百分比
            mean_e = np.mean(e_ratios) * 100
            mean_incorrect = np.mean(incorrect_ratios) * 100
        else:
            mean_correct = 0.0
            mean_e = 0.0
            mean_incorrect = 0.0
        
        summary_data.append({
            "Category": category,
            "Group": group,
            "Model_Size": target_size,
            "Mean_Accuracy_%": round(mean_correct, 2),
            "Mean_E_%": round(mean_e, 2),
            "Mean_Incorrect_%": round(mean_incorrect, 2)
        })

summary_df = pd.DataFrame(summary_data)

# save to CSV
output_csv_path = os.path.join(os.getcwd(), f"{model}/counts/summary_means_{target_size}.csv")
summary_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"Saved summary means to: {output_csv_path}")