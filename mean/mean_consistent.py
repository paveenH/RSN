#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:54:17 2024

@author: paveenhuang

Description:
    This script extracts hidden-state data for samples where 
    the 'None Expert' and 'Expert' roles produce the SAME answer,
    across all 57 tasks, and computes the mean hidden states.
"""

import os
import numpy as np
import json
import argparse

# Task list
TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_medicine",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process model and size arguments for consistent-sample extraction.")
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
args = parser.parse_args()

model = args.model
size = args.size

# Save directories
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
save_path = os.path.join(current_path, "hidden_states_v3_mean", model)
os.makedirs(save_path, exist_ok=True)

# e.g., where the answer JSON files are: "answer/llama3_v3"
json_path = os.path.join(current_path, "answer", f"{model}_v3")

# Lists to store the hidden states across all tasks for consistent samples
all_expert_data = []
all_none_data = []

for task in TASKS:
    try:
        print(f"Processing task: {task}")

        # Construct file paths
        data_expert_filepath = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
        data_none_filepath = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")
        json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")

        # Check if NPY files exist
        if not os.path.exists(data_expert_filepath):
            print(f"Expert hidden-state file not found: {data_expert_filepath}")
            continue
        if not os.path.exists(data_none_filepath):
            print(f"None-expert hidden-state file not found: {data_none_filepath}")
            continue

        # Load NPY data
        data_expert = np.load(data_expert_filepath)     # shape: (samples, 1, layers, hidden_size) or similar
        data_none = np.load(data_none_filepath)

        # Check if JSON file exists
        if not os.path.exists(json_filepath):
            print(f"JSON file not found: {json_filepath}")
            continue

        # Load the data from JSON
        with open(json_filepath, "r", encoding="utf-8") as json_file:
            answer_data = json.load(json_file)

        # Identify consistent samples
        consistent_indices = []
        for idx, entry in enumerate(answer_data.get("data", [])):
            ans_none = entry.get(f"answer_none_{task}")
            ans_expert = entry.get(f"answer_{task}")
            # If the answers are the same, we treat it as consistent
            if ans_none == ans_expert:
                consistent_indices.append(idx)

        if not consistent_indices:
            print(f"No consistent samples found for task: {task}")
            continue

        # Extract data for those consistent indices
        expert_consistent_data = data_expert[consistent_indices, ...]
        none_consistent_data = data_none[consistent_indices, ...]

        # Append to the overall lists
        all_expert_data.append(expert_consistent_data)
        all_none_data.append(none_consistent_data)

        print(f"Processed task: {task}, consistent samples: {len(consistent_indices)}")

    except Exception as e:
        print(f"Error processing task {task}: {e}")

# Combine data across all tasks
if all_expert_data:
    combined_expert = np.concatenate(all_expert_data, axis=0)  # Combine along sample axis
    expert_mean = combined_expert.mean(axis=0, keepdims=True)  # Compute mean across all samples
    expert_mean_filepath = os.path.join(save_path, f"consistent_mean_{size}.npy")
    np.save(expert_mean_filepath, expert_mean)
    print(f"All-expert mean saved to {expert_mean_filepath}")
else:
    print("No expert data found across tasks (no consistent answers).")

if all_none_data:
    combined_none = np.concatenate(all_none_data, axis=0)
    none_mean = combined_none.mean(axis=0, keepdims=True)
    none_mean_filepath = os.path.join(save_path, f"none_consistent_mean_{size}.npy")
    np.save(none_mean_filepath, none_mean)
    print(f"None-expert mean saved to {none_mean_filepath}")
else:
    print("No none-expert data found across tasks (no consistent answers).")