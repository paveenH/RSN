#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates the distribution of correct answers (only A/B/C/D) for each task in MMLU.
Assumes that the correct answers are stored in the "label" field in the sample, with values in the range of 0, 1, 2, 3 corresponding to A, B, C, D.
If the label is not in the [0..3] range, it skips that sample.

Usage example:
  python count_correct_dist.py
"""

import os
import json

# List of tasks (can be adjusted as needed)
TASKS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_medicine",
    "college_mathematics", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous",
    "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

# Mapping from label values to answers
label2ans = {0: "A", 1: "B", 2: "C", 3: "D"}

def main():
    # Current working directory
    current_path = os.getcwd()
    # Directory where JSON files are located (adjust as needed)
    json_root = os.path.join(current_path, "answer", "llama3_v3")

    # Global counts
    global_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    total_samples_global = 0

    # Iterate over each task
    for task in TASKS:
        # Assume JSON file is named as: "{task}_8B_answers.json"
        json_file = os.path.join(json_root, f"{task}_8B_answers.json")
        if not os.path.isfile(json_file):
            print(f"[Warning] JSON file not found for task {task}: {json_file}")
            continue

        # Local task-level counts
        task_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        total_samples_task = 0

        # Read the JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("data", [])

        # Iterate over each sample
        for sample in samples:
            # Get the correct answer from the label field (0 -> A, 1 -> B, 2 -> C, 3 -> D)
            label = sample.get("label")  # If not found, it will be None
            correct_ans = label2ans.get(label)  # If label is not in [0..3], it will be None

            if correct_ans is not None:
                # If the answer is A/B/C/D, increment the count
                task_counts[correct_ans] += 1
                global_counts[correct_ans] += 1

            total_samples_task += 1
            total_samples_global += 1

        # Print statistics for the current task
        print(f"Task: {task}")
        print(f"  Total samples: {total_samples_task}")
        print(f"  A: {task_counts['A']}  B: {task_counts['B']}  C: {task_counts['C']}  D: {task_counts['D']}")
        print("")

    # Global summary statistics for all tasks
    print("=== Global Distribution Across All Tasks ===")
    print(f"Total Samples: {total_samples_global}")
    total_count = sum(global_counts.values())
    print(f"Total Samples with valid ABCD labels: {total_count}")
    print(f"A: {global_counts['A']}")
    print(f"B: {global_counts['B']}")
    print(f"C: {global_counts['C']}")
    print(f"D: {global_counts['D']}")
    print("=============================================")

if __name__ == "__main__":
    main()