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


# Directory containing the JSON files

model = "llama3_v3"  
if model in ["qwen2.5_v4", "phi_v4"]:
    answer_name = "answer_honest"
else:
    answer_name = "answer_honest_revised"
data_dir = os.path.join(os.getcwd(), f"{model}/{answer_name}")
output_dir = os.path.join(os.getcwd(), f"{model}/counts")

# Data storage for statistics
results = defaultdict(lambda: {"none_character": {}, "character": {}})

# Regular expression to extract task and size
# This regex matches filenames like "abstract_algebra_3B_answers.json"
pattern = re.compile(r"^(?P<task>.+)_(?P<size>1B|3B|8B)_answers\.json$")

# ---------------------------
# Read all JSON files and save them to results
# ---------------------------
for file in os.listdir(data_dir):
    if file.endswith("_answers.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            size = match.group("size")
            file_path = os.path.join(data_dir, file)

            # Read the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Replace underscores with spaces for task name to match 'accuracy' keys
            task_name = task.replace('_', ' ')

            # Extract accuracy and performance details
            accuracy = data.get("accuracy", {})
            none_char_key = f"none {task_name}"
            char_key = task_name

            none_char_data = accuracy.get(none_char_key, {})
            char_data = accuracy.get(char_key, {})

            # Store statistics for none-character and character
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
# Aggregate and draw the part
# ---------------------------
category_data = {
    cat: {
        "none_character": defaultdict(lambda: {"correct_ratios": [], 
                                               "e_ratios": [],
                                               "incorrect_ratios": []}),
        "character": defaultdict(lambda: {"correct_ratios": [], 
                                          "e_ratios": [],
                                          "incorrect_ratios": []})
    }
    for cat in categories_map
}

# Find which category a task belongs to
def find_category(task_name):
    """
    Given a task_name, return the category name it belongs to (such as 'STEM', 'Humanities', etc.);
    If not found, return None
    """
    for c, task_list in categories_map.items():
        if task_name in task_list:
            return c
    return None

# Add each task in results to category_data according to its category
for task_name, stats in results.items():
    category = find_category(task_name)
    if category is None:
        print("Task Name Error")
        continue
    
    for size, none_char_info in stats["none_character"].items():
        total_ = none_char_info["total"]
        correct_ = none_char_info["correct"]
        e_ = none_char_info["E_count"]
        if total_ > 0:
            correct_ratio = correct_ / total_
            e_ratio = e_ / total_
            incorrect_ratio = 1.0 - correct_ratio - e_ratio
            
            category_data[category]["none_character"][size]["correct_ratios"].append(correct_ratio)
            category_data[category]["none_character"][size]["e_ratios"].append(e_ratio)
            category_data[category]["none_character"][size]["incorrect_ratios"].append(incorrect_ratio)
    
    for size, char_info in stats["character"].items():
        total_ = char_info["total"]
        correct_ = char_info["correct"]
        e_ = char_info["E_count"]
        if total_ > 0:
            correct_ratio = correct_ / total_
            e_ratio = e_ / total_
            incorrect_ratio = 1.0 - correct_ratio - e_ratio
            
            category_data[category]["character"][size]["correct_ratios"].append(correct_ratio)
            category_data[category]["character"][size]["e_ratios"].append(e_ratio)
            category_data[category]["character"][size]["incorrect_ratios"].append(incorrect_ratio)


for cat in category_data.keys():
    none_char_sizes = sorted(category_data[cat]["none_character"].keys(), key=lambda x: int(x[:-1]))
    char_sizes = sorted(category_data[cat]["character"].keys(), key=lambda x: int(x[:-1]))
    all_sizes = sorted(list(set(none_char_sizes + char_sizes)), key=lambda x: int(x[:-1]))
    
    none_char_correct = []
    none_char_e = []
    none_char_incorrect = []
    
    char_correct = []
    char_e = []
    char_incorrect = []
    
    for size in all_sizes:
        # none_character
        none_data = category_data[cat]["none_character"][size]
        if len(none_data["correct_ratios"]) > 0:
            avg_correct = np.mean(none_data["correct_ratios"])
            avg_e = np.mean(none_data["e_ratios"])
            avg_incorrect = np.mean(none_data["incorrect_ratios"])
        else:
            avg_correct = 0.0
            avg_e = 0.0
            avg_incorrect = 0.0
        
        none_char_correct.append(avg_correct)
        none_char_e.append(avg_e)
        none_char_incorrect.append(avg_incorrect)
        
        # character
        char_data = category_data[cat]["character"][size]
        if len(char_data["correct_ratios"]) > 0:
            avg_correct = np.mean(char_data["correct_ratios"])
            avg_e = np.mean(char_data["e_ratios"])
            avg_incorrect = np.mean(char_data["incorrect_ratios"])
        else:
            avg_correct = 0.0
            avg_e = 0.0
            avg_incorrect = 0.0
        
        char_correct.append(avg_correct)
        char_e.append(avg_e)
        char_incorrect.append(avg_incorrect)
    
    if len(all_sizes) == 0:
        continue
    
    # Plot
    bar_width = 0.35
    r1 = np.arange(len(all_sizes))
    r2 = [x + bar_width for x in r1]
    
    plt.figure(figsize=(8, 4), dpi=100)
        
    # none-character
    h1 = plt.bar(r1, none_char_correct, bar_width, 
                 color='#FFB14E', edgecolor='black', linewidth=1)
    h3 = plt.bar(r1, none_char_e, bar_width, 
                 bottom=np.array(none_char_correct),
                 color='#4E79A7', edgecolor='black', linewidth=1)
    bottom_none = np.array(none_char_correct) + np.array(none_char_e)
    h2 = plt.bar(r1, none_char_incorrect, bar_width, 
                 bottom=bottom_none,
                 color='#4D4D4D', edgecolor='black', linewidth=1)
    
    # character
    h1c = plt.bar(r2, char_correct, bar_width, 
                  color='#FFB14E', edgecolor='black', linewidth=1)
    h3c = plt.bar(r2, char_e, bar_width, 
                  bottom=np.array(char_correct),
                  color='#4E79A7', edgecolor='black', linewidth=1)
    bottom_char = np.array(char_correct) + np.array(char_e)
    h2c = plt.bar(r2, char_incorrect, bar_width, 
                  bottom=bottom_char,
                  color='#4D4D4D', edgecolor='black', linewidth=1)
    
    # x 
    plt.xticks([r + bar_width/2 for r in range(len(all_sizes))], all_sizes)
    for i in range(len(all_sizes)):
        plt.text(r1[i], -0.05, 'none-char', ha='center', va='top', rotation=45)
        plt.text(r2[i], -0.05, 'char', ha='center', va='top', rotation=45)
    
    # axis and label
    # plt.xlabel('Model Size', labelpad=20)
    plt.ylabel('Proportion of Questions') 
    plt.title(f'{cat}', pad=20)
    
    plt.legend([h2, h3, h1], 
               ['Incorrect (non-E)', 'E Responses', 'Correct'], 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(False)
    
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    cat_plot_file = os.path.join(output_dir, f"{cat}_category_performance.png")
    print(f"save to {cat_plot_file}")
    plt.savefig(cat_plot_file, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()