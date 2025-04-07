#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:53:09 2025

@author: paveenhuang
"""
import os
import numpy as np
# import argparse

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
# parser = argparse.ArgumentParser(description="Process model and size arguments.")
# parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
# parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
# args = parser.parse_args()

model = "llama3"
size = "8B"

# Set directories
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
save_path = os.path.join(current_path, "hidden_states_v3_mean", model)
os.makedirs(save_path, exist_ok=True)


# Initialize lists to store data across tasks
all_char_data = []
all_none_char_data = []

for task in TASKS:
    try:
        print(f"Processing task: {task}")

        # Construct file paths for Expert (char) and None-Expert (none_char)
        data_char_filepath = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
        data_none_char_filepath = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")

        # Check if NPY files exist
        if not os.path.exists(data_char_filepath):
            print(f"Expert data file not found: {data_char_filepath}")
            continue
        if not os.path.exists(data_none_char_filepath):
            print(f"None-expert data file not found: {data_none_char_filepath}")
            continue

        # Load NPY data
        data_char = np.load(data_char_filepath)
        data_none_char = np.load(data_none_char_filepath)

        # Append to overall lists
        all_char_data.append(data_char)
        all_none_char_data.append(data_none_char)

        print(f"Processed task: {task}, samples: {data_char.shape[0]}")

    except Exception as e:
        print(f"Error processing task {task}: {e}")

# Combine data across all tasks and compute mean hidden states
if all_char_data:
    combined_char = np.concatenate(all_char_data, axis=0)  # Combine along sample axis
    mean_char = combined_char.mean(axis=0, keepdims=True)    # Mean over all samples
    char_mean_filepath = os.path.join(save_path, f"all_mean_{size}.npy")
    np.save(char_mean_filepath, mean_char)
    print(f"Expert mean hidden states saved to {char_mean_filepath}")
else:
    print("No expert data found across tasks.")

if all_none_char_data:
    combined_none = np.concatenate(all_none_char_data, axis=0)
    mean_none = combined_none.mean(axis=0, keepdims=True)
    none_mean_filepath = os.path.join(save_path, f"none_all_mean_{size}.npy")
    np.save(none_mean_filepath, mean_none)
    print(f"None-expert mean hidden states saved to {none_mean_filepath}")
else:
    print("No none-expert data found across tasks.")
