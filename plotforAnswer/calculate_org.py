#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:06:55 2024

Author: paveenhuang
"""

import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import re
import pandas as pd

# Define tasks for different domains
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

# Specify model, answer_name, and size (only the specified size data is plotted)
model = "llama3_v3"  

size = "8B"  # Specify size

# answer_name = "answer_student_revised"
# key1 = "beginner"
# key2 = "advanced"

answer_name = "answer_honest_revised"
key1 = "none"
key2 = ""

data_dir = os.path.join(os.getcwd(), f"{model}/{answer_name}")
output_dir = os.path.join(os.getcwd(), f"{model}/counts")

# Data storage structure
results = defaultdict(lambda: {key1: {}, key2: {}})

# Regular expression to match filenames, e.g., "abstract_algebra_3B_answers.json"
pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3.8B|7B|8B)_answers(_\d+)?\.json$")

# ---------------------------
# Read JSON files and store data into results
# ---------------------------
for file in os.listdir(data_dir):
    if file.endswith("_answers.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            file_size = match.group("size")
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            task_name = task.replace('_', ' ')
            accuracy = data.get("accuracy", {})
            
            none_char_key = f"none {task_name}"
            char_key = task_name
            
            # none_char_key = f"{key1} {task_name}"
            # char_key = f"{key2} {task_name}"

            none_char_data = accuracy.get(none_char_key, {})
            char_data = accuracy.get(char_key, {})

            results[task_name][key1][file_size] = {
                "correct": none_char_data.get("correct", 0),
                "total": none_char_data.get("total", 0),
                "E_count": none_char_data.get("E_count", 0),
            }
            results[task_name][key2][file_size] = {
                "correct": char_data.get("correct", 0),
                "total": char_data.get("total", 0),
                "E_count": char_data.get("E_count", 0),
            }
        else:
            print(f"Filename does not match pattern and will be skipped: {file}")

# ---------------------------
# Aggregate data: Summarize data for the specified size across domains
# ---------------------------
# Initialize domain_data, storing None-Expert and Expert data for each domain
category_data = {
    cat: {
        key1: defaultdict(lambda: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []}),
        key2: defaultdict(lambda: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []})
    }
    for cat in categories_map
}

def find_category(task_name):
    for cat, task_list in categories_map.items():
        if task_name in task_list:
            return cat
    return None

for task_name, stats in results.items():
    category = find_category(task_name)
    if category is None:
        print("Task Name Error:", task_name)
        continue

    none_info = stats[key1].get(size, None)
    char_info = stats[key2].get(size, None)

    if none_info is not None and none_info["total"] > 0:
        total_ = none_info["total"]
        correct_ratio = none_info["correct"] / total_
        e_ratio = none_info["E_count"] / total_
        incorrect_ratio = 1.0 - correct_ratio - e_ratio
        category_data[category][key1][size]["correct_ratios"].append(correct_ratio)
        category_data[category][key1][size]["e_ratios"].append(e_ratio)
        category_data[category][key1][size]["incorrect_ratios"].append(incorrect_ratio)

    if char_info is not None and char_info["total"] > 0:
        total_ = char_info["total"]
        correct_ratio = char_info["correct"] / total_
        e_ratio = char_info["E_count"] / total_
        incorrect_ratio = 1.0 - correct_ratio - e_ratio
        category_data[category][key2][size]["correct_ratios"].append(correct_ratio)
        category_data[category][key2][size]["e_ratios"].append(e_ratio)
        category_data[category][key2][size]["incorrect_ratios"].append(incorrect_ratio)


# ---------------------------
# Compute average ratios for each domain (only for the specified size)
# ---------------------------
domains = list(categories_map.keys())
none_correct, none_e, none_incorrect = [], [], []
char_correct, char_e, char_incorrect = [], [], []

for domain in domains:
    data_none = category_data[domain][key1].get(size, None)
    data_char = category_data[domain][key2].get(size, None)

    if data_none is not None and data_none["correct_ratios"]:
        avg_none_correct = np.mean(data_none["correct_ratios"])
        avg_none_e = np.mean(data_none["e_ratios"])
        avg_none_incorrect = np.mean(data_none["incorrect_ratios"])
    else:
        avg_none_correct = avg_none_e = avg_none_incorrect = 0.0

    none_correct.append(avg_none_correct)
    none_e.append(avg_none_e)
    none_incorrect.append(avg_none_incorrect)

    if data_char is not None and data_char["correct_ratios"]:
        avg_char_correct = np.mean(data_char["correct_ratios"])
        avg_char_e = np.mean(data_char["e_ratios"])
        avg_char_incorrect = np.mean(data_char["incorrect_ratios"])
    else:
        avg_char_correct = avg_char_e = avg_char_incorrect = 0.0

    char_correct.append(avg_char_correct)
    char_e.append(avg_char_e)
    char_incorrect.append(avg_char_incorrect)

# ---------------------------
# Plotting: Display data for all four domains on the same graph,
# ---------------------------
plt.figure(figsize=(12, 7), dpi=300)

# Keep the original color scheme
colors = {
    key1: {
        'correct': '#FFD48E',    # Light orange
        'e': '#7DA1C4',         # Light blue
        'incorrect': '#808080'   # Light gray
    },
    key2: {
        'correct': '#FF8C00',    # Dark orange
        'e': '#2E5A88',         # Dark blue
        'incorrect': '#404040'   # Dark gray
    }
}

bar_width = 0.35
x = np.arange(len(domains))  # 4 domain groups
r1 = x - bar_width/2         # Position for None-Expert bars
r2 = x + bar_width/2         # Position for Expert bars

# Plot None-Expert stacked bars
plt.bar(r1, none_correct, bar_width, color=colors[key1]['correct'], edgecolor='black', linewidth=1.2, label=f'Correct ({key1})')
plt.bar(r1, none_e, bar_width, bottom=np.array(none_correct), color=colors[key1]['e'], edgecolor='black', linewidth=1.2, label=f'E Responses ({key1})')
plt.bar(r1, none_incorrect, bar_width, bottom=np.array(none_correct)+np.array(none_e), color=colors[key1]['incorrect'], edgecolor='black', linewidth=1.2, label=f'Incorrect ({key1})')

# Plot Expert stacked bars
plt.bar(r2, char_correct, bar_width, color=colors[key2]['correct'], edgecolor='black', linewidth=1.2, label=f'Correct ({key2})')
plt.bar(r2, char_e, bar_width, bottom=np.array(char_correct), color=colors[key2]['e'], edgecolor='black', linewidth=1.2, label=f'E Responses ({key2})')
plt.bar(r2, char_incorrect, bar_width, bottom=np.array(char_correct)+np.array(char_e), color=colors[key2]['incorrect'], edgecolor='black', linewidth=1.2, label=f'Incorrect ({key2})')

# Add horizontal grid lines for better readability
plt.gca().yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
plt.gca().xaxis.grid(False)

plt.xticks(x, domains, fontsize=12)
plt.ylabel("Proportion of Answers", fontsize=14, fontweight='bold')
plt.title(f"Performance for {model[:-3]} ({size}) across Domains", fontsize=16, fontweight='bold', pad=20)

# Merge
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), fontsize=8, frameon=True, edgecolor='black')

# Axis
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

plt.tight_layout()

# Save
output_file = os.path.join(output_dir, f"all_domains_{size}_performance.png")
print(f"Saving plot to {output_file}")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()


# ---------------------------
# 10) Save results to CSV
# ---------------------------
# Prepare data storage
csv_data = []
headers = ["Group"]
categories = list(categories_map.keys())
group_names = ["none", "",]
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
csv_output_path = os.path.join(output_dir, f"performance_comparison_{size}.csv")

df.to_csv(csv_output_path, index=False)
print(f"Saved CSV file to {csv_output_path}")

# Display the DataFrame for quick verification
import ace_tools as tools
tools.display_dataframe_to_user(name="Performance Comparison", dataframe=df)