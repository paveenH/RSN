#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:54:17 2024

@author: paveenhuang
"""

## Calculate Mean for ALl Tasks

import os
import numpy as np
import json

# Task list
TASKS = [
    "abstract_algebra", "anatomy", "astronomy", 
    # "business_ethics", "clinical_knowledge", 
    # "college_biology", "college_chemistry", "college_computer_science", "college_medicine", 
    # "college_mathematics", "college_physics", "computer_security", "conceptual_physics", 
    # "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", 
    # "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", 
    # "high_school_european_history", "high_school_geography", "high_school_government_and_politics", 
    # "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", 
    # "high_school_physics", "high_school_psychology", "high_school_statistics", 
    # "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", 
    # "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", 
    # "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", 
    # "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", 
    # "professional_medicine", "professional_psychology", "public_relations", "security_studies", 
    # "sociology", "us_foreign_policy", "virology", "world_religions"
]

# Save directory
path = os.getcwd()
save = path + "/hidden"
json_path = path + "/answer_honest_an"

# Size of model
size = "8B"

# Initialize lists to store data across tasks
all_char_diff_data = []
all_none_char_diff_data = []

for task in TASKS:
    try:
        # Load task-specific data
        data_char = np.load(f'{save}/{task}_{task}_{size}.npy')  
        data_none_char = np.load(f'{save}/none_{task}_{task}_{size}.npy')  

        # Load inconsistent indices from JSON
        json_filepath = f'{json_path}/{task}_{size}_answers.json'
        with open(json_filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        inconsistent_indices = []
        # answer_mapping = ["A", "B", "C", "D", "E"]
        for idx, entry in enumerate(data.get("data", []), start=0):
            ans_none = entry.get(f"answer_none_{task}")
            ans_abst = entry.get(f"answer_{task}")
            if ans_none != ans_abst:
                inconsistent_indices.append(idx)

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
    np.save(f"{save}/all_mean_{size}.npy", char_mean)
    print(f"All char mean saved to {save}/all_mean_{size}.npy")
else:
    print("No char differences found across tasks.")

if all_none_char_diff_data:
    combined_none_char_diff = np.concatenate(all_none_char_diff_data, axis=0)  # Combine along sample axis
    none_char_mean = combined_none_char_diff.mean(axis=0, keepdims=True)  # Compute mean across all samples
    np.save(f"{save}/none_all_mean_{size}.npy", none_char_mean)
    print(f"none-char mean saved to {save}/none_all_mean_{size}.npy")
else:
    print("No none-char differences found across tasks.")