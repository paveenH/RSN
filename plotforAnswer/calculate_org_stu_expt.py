#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 17:31:14 2025

@author: paveenhuang
"""

import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import re

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

student_answer_name = "answer_student_revised"
expert_answer_name = "answer_honest_revised"

expert_dir = os.path.join(os.getcwd(), f"{model}/{expert_answer_name}")
student_dir = os.path.join(os.getcwd(), f"{model}/{student_answer_name}")
output_dir = os.path.join(os.getcwd(), f"{model}/counts")

key_none = "none"
key_beginner = "beginner"
key_advanced = "advanced"
key_expert = "expert"


# Data storage structure: Store data for 4 roles under each task
results = defaultdict(lambda: {key_none: {}, key_beginner: {}, key_advanced: {}, key_expert: {}})

# Regular expression to match filenames, e.g., "abstract_algebra_3B_answers.json"
pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers(_\d+)?\.json$")

# ---------------------------
# Read Expert Data (expert_dir), store none and expert data
# ---------------------------
for file in os.listdir(expert_dir):
    if file.endswith(f"{size}_answers.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            file_path = os.path.join(expert_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            task_name = task.replace('_', ' ')
            accuracy = data.get("accuracy", {})

            # In the Expert directory, assuming JSON contains keys:
            #   "none {task_name}" for None Expert data
            #   "{task_name}" for Expert data
            none_key = f"{key_none} {task_name}"  # e.g., "none abstract algebra"
            expert_key = task_name                # Expert key is the task name

            none_data = accuracy.get(none_key, {})
            expert_data = accuracy.get(expert_key, {})

            results[task_name][key_none] = {
                "correct": none_data.get("correct", 0),
                "total": none_data.get("total", 0),
                "E_count": none_data.get("E_count", 0)
            }
            results[task_name][key_expert] = {
                "correct": expert_data.get("correct", 0),
                "total": expert_data.get("total", 0),
                "E_count": expert_data.get("E_count", 0)
            }
        else:
            print(f"Filename does not match pattern and will be skipped: {file}")

# ---------------------------
# Read Student Data (student_dir), store beginner and advanced data
# ---------------------------
for file in os.listdir(student_dir):
    if file.endswith(f"{size}_answers.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            file_path = os.path.join(student_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            task_name = task.replace('_', ' ')
            accuracy = data.get("accuracy", {})

            beginner_key_full = f"{key_beginner} {task_name}"
            advanced_key_full = f"{key_advanced} {task_name}"

            beginner_data = accuracy.get(beginner_key_full, {})
            advanced_data = accuracy.get(advanced_key_full, {})

            results[task_name][key_beginner] = {
                "correct": beginner_data.get("correct", 0),
                "total": beginner_data.get("total", 0),
                "E_count": beginner_data.get("E_count", 0)
            }
            results[task_name][key_advanced] = {
                "correct": advanced_data.get("correct", 0),
                "total": advanced_data.get("total", 0),
                "E_count": advanced_data.get("E_count", 0)
            }
        else:
            print(f"Filename does not match pattern and will be skipped: {file}")

# After the aggregation, compute average ratios and plot using results
# where results[task_name] contains four roles: key_none, key_beginner, key_advanced, key_expert

print("JSON files have been read and stored.")

# ---------------------------
# Aggregate Data: Summarize data for specified size
# ---------------------------
category_data = {
    cat: {
        key_none: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []},
        key_beginner: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []},
        key_advanced: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []},
        key_expert: {"correct_ratios": [], "e_ratios": [], "incorrect_ratios": []}
    }
    for cat in categories_map
}

def find_category(task_name):
    for cat, task_list in categories_map.items():
        if task_name in task_list:
            return cat
    return None

for task_name, stats_dict in results.items():
    category = find_category(task_name)
    if category is None:
        print("Task Name Error:", task_name)
        continue

    info_none = stats_dict[key_none]
    info_beginner = stats_dict[key_beginner]
    info_advanced = stats_dict[key_advanced]
    info_expert = stats_dict[key_expert]

    if info_none is not None and info_none["total"] > 0:
        total = info_none["total"]
        corr_ratio = info_none["correct"] / total
        e_ratio = info_none["E_count"] / total
        inc_ratio = 1.0 - corr_ratio - e_ratio
        category_data[category][key_none]["correct_ratios"].append(corr_ratio)
        category_data[category][key_none]["e_ratios"].append(e_ratio)
        category_data[category][key_none]["incorrect_ratios"].append(inc_ratio)
    if info_beginner is not None and info_beginner["total"] > 0:
        total = info_beginner["total"]
        corr_ratio = info_beginner["correct"] / total
        e_ratio = info_beginner["E_count"] / total
        inc_ratio = 1.0 - corr_ratio - e_ratio
        category_data[category][key_beginner]["correct_ratios"].append(corr_ratio)
        category_data[category][key_beginner]["e_ratios"].append(e_ratio)
        category_data[category][key_beginner]["incorrect_ratios"].append(inc_ratio)
    if info_advanced is not None and info_advanced["total"] > 0:
        total = info_advanced["total"]
        corr_ratio = info_advanced["correct"] / total
        e_ratio = info_advanced["E_count"] / total
        inc_ratio = 1.0 - corr_ratio - e_ratio
        category_data[category][key_advanced]["correct_ratios"].append(corr_ratio)
        category_data[category][key_advanced]["e_ratios"].append(e_ratio)
        category_data[category][key_advanced]["incorrect_ratios"].append(inc_ratio)
    if info_expert is not None and info_expert["total"] > 0:
        total = info_expert["total"]
        corr_ratio = info_expert["correct"] / total
        e_ratio = info_expert["E_count"] / total
        inc_ratio = 1.0 - corr_ratio - e_ratio
        category_data[category][key_expert]["correct_ratios"].append(corr_ratio)
        category_data[category][key_expert]["e_ratios"].append(e_ratio)
        category_data[category][key_expert]["incorrect_ratios"].append(inc_ratio)

# ---------------------------
# Compute average ratios for each domain (for the specified size)
# ---------------------------
domains = list(categories_map.keys())  # e.g., ["STEM", "Humanities", "Social Sciences", "Other"]

# Initialize average correct, E, incorrect ratios for each role and domain
avg_data = {domain: {key_none: {}, key_beginner: {}, key_advanced: {}, key_expert: {}} for domain in domains}

for domain in domains:
    for role in [key_none, key_beginner, key_advanced, key_expert]:
        data_role = category_data[domain][role]
        if data_role is not None and data_role["correct_ratios"]:
            avg_data[domain][role]["correct"] = np.mean(data_role["correct_ratios"])
            avg_data[domain][role]["e"] = np.mean(data_role["e_ratios"])
            avg_data[domain][role]["incorrect"] = np.mean(data_role["incorrect_ratios"])
        else:
            avg_data[domain][role]["correct"] = 0.0
            avg_data[domain][role]["e"] = 0.0
            avg_data[domain][role]["incorrect"] = 0.0

# Store average ratios for each role across all domains
none_correct = []
none_e = []
none_incorrect = []
begin_correct = []
begin_e = []
begin_incorrect = []
adv_correct = []
adv_e = []
adv_incorrect = []
exp_correct = []
exp_e = []
exp_incorrect = []

for domain in domains:
    none_correct.append(avg_data[domain][key_none]["correct"])
    none_e.append(avg_data[domain][key_none]["e"])
    none_incorrect.append(avg_data[domain][key_none]["incorrect"])
    
    begin_correct.append(avg_data[domain][key_beginner]["correct"])
    begin_e.append(avg_data[domain][key_beginner]["e"])
    begin_incorrect.append(avg_data[domain][key_beginner]["incorrect"])
    
    adv_correct.append(avg_data[domain][key_advanced]["correct"])
    adv_e.append(avg_data[domain][key_advanced]["e"])
    adv_incorrect.append(avg_data[domain][key_advanced]["incorrect"])
    
    exp_correct.append(avg_data[domain][key_expert]["correct"])
    exp_e.append(avg_data[domain][key_expert]["e"])
    exp_incorrect.append(avg_data[domain][key_expert]["incorrect"])

# ---------------------------
# Plot: Draw 4 stacked bars for each domain, showing the proportions of Correct, E, Incorrect for each role
# ---------------------------
plt.figure(figsize=(12, 7), dpi=300)

# Define colors
colors = {
    key_none: {
        'correct': '#FFD48E',    # Light orange
        'e': '#7DA1C4',         # Light blue
        'incorrect': '#808080'   # Gray
    },
    key_beginner: {
        'correct': '#FFB6C1',    # Light pink
        'e': '#87CEFA',         # Sky blue
        'incorrect': '#A9A9A9'    # Dark gray
    },
    key_advanced: {
        'correct': '#FFA07A',    # Light salmon
        'e': '#6495ED',         # Cornflower blue
        'incorrect': '#696969'   # Dark gray
    },
    key_expert: {
        'correct': '#FF8C00',    # Dark orange
        'e': '#2E5A88',         # Dark blue
        'incorrect': '#404040'   # Dark gray
    }
}

bar_width = 0.20
x = np.arange(len(domains))  # 4 domains
# Calculate x-axis offset for each role (4 bars per group, centered at x)
r_none = x - 1.5 * bar_width
r_begin = x - 0.5 * bar_width
r_adv = x + 0.5 * bar_width
r_exp = x + 1.5 * bar_width

# Plot None Expert stacked bars
plt.bar(r_none, none_correct, bar_width, color=colors[key_none]['correct'], edgecolor='black', linewidth=1.2, label=f'Correct ({key_none})')
plt.bar(r_none, none_e, bar_width, bottom=np.array(none_correct), color=colors[key_none]['e'], edgecolor='black', linewidth=1.2, label=f'E Responses ({key_none})')
plt.bar(r_none, none_incorrect, bar_width, bottom=np.array(none_correct)+np.array(none_e), color=colors[key_none]['incorrect'], edgecolor='black', linewidth=1.2, label=f'Incorrect ({key_none})')

# Plot Beginner Student stacked bars
plt.bar(r_begin, begin_correct, bar_width, color=colors[key_beginner]['correct'], edgecolor='black', linewidth=1.2, label=f'Correct ({key_beginner})')
plt.bar(r_begin, begin_e, bar_width, bottom=np.array(begin_correct), color=colors[key_beginner]['e'], edgecolor='black', linewidth=1.2, label=f'E Responses ({key_beginner})')
plt.bar(r_begin, begin_incorrect, bar_width, bottom=np.array(begin_correct)+np.array(begin_e), color=colors[key_beginner]['incorrect'], edgecolor='black', linewidth=1.2, label=f'Incorrect ({key_beginner})')

# Plot Advanced Student stacked bars
plt.bar(r_adv, adv_correct, bar_width, color=colors[key_advanced]['correct'], edgecolor='black', linewidth=1.2, label=f'Correct ({key_advanced})')
plt.bar(r_adv, adv_e, bar_width, bottom=np.array(adv_correct), color=colors[key_advanced]['e'], edgecolor='black', linewidth=1.2, label=f'E Responses ({key_advanced})')
plt.bar(r_adv, adv_incorrect, bar_width, bottom=np.array(adv_correct)+np.array(adv_e), color=colors[key_advanced]['incorrect'], edgecolor='black', linewidth=1.2, label=f'Incorrect ({key_advanced})')

# Plot Expert stacked bars
plt.bar(r_exp, exp_correct, bar_width, color=colors[key_expert]['correct'], edgecolor='black', linewidth=1.2, label=f'Correct ({key_expert})')
plt.bar(r_exp, exp_e, bar_width, bottom=np.array(exp_correct), color=colors[key_expert]['e'], edgecolor='black', linewidth=1.2, label=f'E Responses ({key_expert})')
plt.bar(r_exp, exp_incorrect, bar_width, bottom=np.array(exp_correct)+np.array(exp_e), color=colors[key_expert]['incorrect'], edgecolor='black', linewidth=1.2, label=f'Incorrect ({key_expert})')

plt.xticks(x, domains, fontsize=12)
plt.ylabel("Proportion of Answers", fontsize=14, fontweight='bold')
plt.title(f"Performance for {model[:-3]} ({size}) across Domains", fontsize=16, fontweight='bold', pad=20)

# Combine legends (deduplicate)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), fontsize=8, frameon=True, edgecolor='black')

# Set axis borders
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.2)

plt.gca().yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
plt.gca().xaxis.grid(False)
plt.tight_layout()

# Save the plot
output_file = os.path.join(output_dir, f"all_domains_{size}_performance.png")
print(f"Saving plot to {output_file}")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"Plot saved to {output_file}")