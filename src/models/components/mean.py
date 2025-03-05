#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:54:17 2024

@author: paveenhuang
"""

## Calculate Mean for All Tasks

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
parser = argparse.ArgumentParser(description="Process model and size arguments.")
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
args = parser.parse_args()

model = args.model
size = args.size

# Save directories
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v5", model)
save_path = os.path.join(current_path, "hidden_states_v5_mean", model)
os.makedirs(save_path, exist_ok=True)

json_path = os.path.join(current_path, "answer", f"{model}_v5")

# Initialize lists to store data across tasks
all_char_diff_data = []
all_none_char_diff_data = []

for task in TASKS:
    try:
        print(f"Processing task: {task}")

        # Construct file paths
        data_char_filepath = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
        data_none_char_filepath = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")
        json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")

        # Check if NPY files exist
        if not os.path.exists(data_char_filepath):
            print(f"Data char file not found: {data_char_filepath}")
            continue
        if not os.path.exists(data_none_char_filepath):
            print(f"Data none-char file not found: {data_none_char_filepath}")
            continue

        # Load NPY data
        data_char = np.load(data_char_filepath)
        data_none_char = np.load(data_none_char_filepath)

        # Check if JSON file exists
        if not os.path.exists(json_filepath):
            print(f"JSON file not found: {json_filepath}")
            continue

        # Load inconsistent indices from JSON
        with open(json_filepath, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        inconsistent_indices = []
        for idx, entry in enumerate(data.get("data", [])):
            ans_none = entry.get(f"answer_none_{task}")
            ans_abst = entry.get(f"answer_{task}")
            if ans_none != ans_abst:
                inconsistent_indices.append(idx)

        if not inconsistent_indices:
            print(f"No inconsistent samples found for task: {task}")
            continue

        # Extract data for inconsistent indices
        data_char_diff = data_char[inconsistent_indices, ...]
        data_none_char_diff = data_none_char[inconsistent_indices, ...]

        # Append to overall lists
        all_char_diff_data.append(data_char_diff)
        all_none_char_diff_data.append(data_none_char_diff)

        print(f"Processed task: {task}, inconsistent samples: {len(inconsistent_indices)}")

    except Exception as e:
        print(f"Error processing task {task}: {e}")

# Combine data across all tasks
if all_char_diff_data:
    combined_char_diff = np.concatenate(all_char_diff_data, axis=0)  # Combine along sample axis
    char_mean = combined_char_diff.mean(axis=0, keepdims=True)  # Compute mean across all samples
    char_mean_filepath = os.path.join(save_path, f"all_mean_{size}.npy")
    np.save(char_mean_filepath, char_mean)
    print(f"All char mean saved to {char_mean_filepath}")
else:
    print("No char differences found across tasks.")

if all_none_char_diff_data:
    combined_none_char_diff = np.concatenate(all_none_char_diff_data, axis=0)  # Combine along sample axis
    none_char_mean = combined_none_char_diff.mean(axis=0, keepdims=True)  # Compute mean across all samples
    none_char_mean_filepath = os.path.join(save_path, f"none_all_mean_{size}.npy")
    np.save(none_char_mean_filepath, none_char_mean)
    print(f"None-char mean saved to {none_char_mean_filepath}")
else:
    print("No none-char differences found across tasks.")
