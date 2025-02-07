#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:43:46 2025

Script to extract detailed 'invalid' counts and other metrics from JSON files and save them into separate CSV files per size.

Author: paveenhuang
"""

import os
import json
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# =======================
# Configuration
# =======================
model = "qwen2.5_v4"  
top = 17
alpha = 3
answer_name = f"answer_modified_alpha{alpha}_revised"
data_dir = os.path.join(os.getcwd(), f"{model}/{answer_name}")
output_dir = os.path.join(os.getcwd(), f"{model}/counts")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize a nested defaultdict to store detailed metrics per size and task
# Structure: csv_data_per_size[size][task] = {metrics}
csv_data_per_size = defaultdict(lambda: defaultdict(lambda: {
    "task": "",
    "size": "",
    "correct_num_non": 0,
    "correct_num_char": 0,
    "non_char_acc": 0.0,
    "char_acc": 0.0,
    "none_char_E": 0,
    "char_E": 0,
    "non_invalid": 0,
    "char_invalid": 0,
    "total": 0
}))

# This regex matches filenames like "abstract_algebra_3B_answers.json"
# pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3.8B|7B|8B)_answers\.json$")
pattern = re.compile(r"^(?P<task>.+)_(?P<size>0\.5B|1B|3B|3\.8B|7B|8B)_answers(_\d+)?\.json$")


# =======================
# Processing JSON Files
# =======================
for file in os.listdir(data_dir):
    if file.endswith(f"_answers_{top}.json"):
        match = pattern.match(file)
        if match:
            task = match.group("task")
            size = match.group("size")
            file_path = os.path.join(data_dir, file)

            # Read the JSON file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file}: {e}")
                continue

            # Replace underscores with spaces for task name to match 'accuracy' keys
            task_name = task.replace('_', ' ')
            accuracy_dict = data["accuracy"]

            for character_type, metrics in accuracy_dict.items():
                invalid_count = metrics.get("invalid", 0)
                correct = metrics.get("correct", 0)
                accuracy_percentage = metrics.get("accuracy_percentage", 0.0)
                E_count = metrics.get("E_count", 0)
                total = metrics.get("total", 0)

                # Initialize task and size if not already
                csv_data_per_size[size][task_name]["task"] = task_name
                csv_data_per_size[size][task_name]["size"] = size

                if character_type.startswith("none "):
                    # 非字符类型
                    csv_data_per_size[size][task_name]["correct_num_non"] = correct
                    csv_data_per_size[size][task_name]["non_char_acc"] = accuracy_percentage
                    csv_data_per_size[size][task_name]["none_char_E"] = E_count
                    csv_data_per_size[size][task_name]["non_invalid"] = invalid_count
                else:
                    # 字符类型
                    csv_data_per_size[size][task_name]["correct_num_char"] = correct
                    csv_data_per_size[size][task_name]["char_acc"] = accuracy_percentage
                    csv_data_per_size[size][task_name]["char_E"] = E_count
                    csv_data_per_size[size][task_name]["char_invalid"] = invalid_count

                # Calculate total invalid
            csv_data_per_size[size][task_name]["total"] = total

# =======================
# Write Separate CSV Files per Size
# =======================
csv_headers = [
    "task",
    "size",
    "correct_num_non",
    "correct_num_char",
    "non_char_acc",
    "char_acc",
    "none_char_E",
    "char_E",
    "non_invalid",
    "char_invalid",
    "total"
]

for size, tasks in csv_data_per_size.items():
    # Define the output CSV path for the current size
    output_csv = os.path.join(output_dir, f"total_counts_mdf_{size}.csv")

    # Prepare data for CSV
    csv_rows = []
    for task, metrics in tasks.items():
        row = {
            "task": metrics["task"],
            "size": metrics["size"],
            "correct_num_non": metrics["correct_num_non"],
            "correct_num_char": metrics["correct_num_char"],
            "non_char_acc": metrics["non_char_acc"],
            "char_acc": metrics["char_acc"],
            "none_char_E": metrics["none_char_E"],
            "char_E": metrics["char_E"],
            "non_invalid": metrics["non_invalid"],
            "char_invalid": metrics["char_invalid"],
            "total": metrics["total"]
        }
        csv_rows.append(row)

    # Write the records to the CSV file
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"save 'total counts' to: {output_csv}")

# =======================
# Visualization (Combined)
# =======================

for size, tasks in csv_data_per_size.items():
    # Create a DataFrame from the tasks dictionary
    df = pd.DataFrame([
        {
            "task": metrics["task"],
            "correct_num_non": metrics["correct_num_non"],
            "correct_num_char": metrics["correct_num_char"],
            "non_char_acc": metrics["non_char_acc"],
            "char_acc": metrics["char_acc"],
            "none_char_E": metrics["none_char_E"],
            "char_E": metrics["char_E"],
            "non_invalid": metrics["non_invalid"],
            "char_invalid": metrics["char_invalid"],
            "total": metrics["total"]
        }
        for task, metrics in tasks.items()
    ])
    
    # Sort the DataFrame by 'task' in alphabetical order
    df = df.sort_values(by='task').reset_index(drop=True)

    # Prepare data for plotting
    index = range(len(df))
    bar_width = 0.35
    df["none_char_E_pct"] = (df["none_char_E"] / df["total"]) * 100
    df["char_E_pct"] = (df["char_E"] / df["total"]) * 100

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 2]}, sharex=True)
    fig.subplots_adjust(hspace=0.05)  # Adjust space between subplots

    # Plot Accuracy Percentage (Top Subplot)
    h1 = ax1.bar(index, df["non_char_acc"], bar_width, label='Non-Character Accuracy', color='lightgreen')
    h2 = ax1.bar([i + bar_width for i in index], df["char_acc"], bar_width, label='Character Accuracy', color='salmon')
    ax1.set_ylabel("Accuracy Percentage")
    ax1.set_title(f"Accuracy and E Counts per Task {model} ({size})")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot E Counts Percentage (Bottom Subplot, inverted)
    h3 = ax2.bar(index, df["none_char_E_pct"], bar_width, label='Non-Character E Counts', color='gold')
    h4 = ax2.bar([i + bar_width for i in index], df["char_E_pct"], bar_width, label='Character E Counts', color='pink')
    ax2.set_ylabel("E Counts (%)")
    ax2.invert_yaxis()  # Invert the y-axis for the E Counts
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Set x-axis ticks and labels for both plots
    ax2.set_xticks([i + bar_width / 2 for i in index])
    ax2.set_xticklabels(df["task"], rotation=90)

    # Add a single legend for the entire figure in the top-right corner
    fig.legend(
        [h1, h2, h3, h4],
        ['Non-Character-mdf Accuracy', 'Character-mdf Accuracy', 'Non-Character-mdf E Counts', 'Character-mdf E Counts'],
        loc='upper center',  # Position in the top-right corner
        bbox_to_anchor=(1.1, 0.98),  # Slightly outside the plot area
        fontsize=10
    )

    # Save the combined plot
    combined_plot_path = os.path.join(output_dir, f"combined_accuracy_E_counts_mdf_{size}.png")
    plt.tight_layout()
    plt.savefig(combined_plot_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Done combined plot: {combined_plot_path}")

print("Done")

