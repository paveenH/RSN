#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:54:17 2024
@description:
    Mean NMD for Expert - NoExpert in diff answer-pairs

@author: paveenhuang
"""

## Calculate Mean for All Tasks

import os
import numpy as np
import json
from task_list import TASKS

# ==================== Configuration ====================
model = "llama3"
size = "70B"
TYPE = "non"

# Base directory (NCHC or local)
DIR = "/work/d12922004/RolePlaying/components"
# DIR = "/data2/paveen/RolePlaying/components"  # Local alternative

# Paths: JSON in answer_non/llama3/, hidden states in hidden_states_non/llama3/
json_path = os.path.join(DIR, f"answer_{TYPE}", model)
hidden_states_path = os.path.join(DIR, f"hidden_states_{TYPE}", model)

save_path = os.path.join(DIR, "hidden_states_mean", f"{model}_{TYPE}")
os.makedirs(save_path, exist_ok=True)
print("save path: ", save_path)

# Initialize lists to store data across tasks
all_char_diff_data = []
all_none_char_diff_data = []

for task in TASKS:
    try:
        print(f"Processing task: {task}")

        # Construct file paths
        # File naming: {task}_expert_{task}_{size}.npy, non_{task}_expert_{task}_{size}.npy
        data_char_filepath = os.path.join(hidden_states_path, f"{task}_expert_{task}_{size}.npy")
        data_none_char_filepath = os.path.join(hidden_states_path, f"{TYPE}_{task}_expert_{task}_{size}.npy")
        json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")

        # Check if NPY files exist
        if not os.path.exists(data_char_filepath):
            print(f"Data char file not found: {data_char_filepath}")
            continue
        if not os.path.exists(data_none_char_filepath):
            print(f"Data none-char file not found: {data_none_char_filepath}")
            continue

        # Extract data for inconsistent indices
        data_char = np.load(data_char_filepath)
        data_none_char = np.load(data_none_char_filepath)
        
        # Ensure both are in shape (N, 1, 33, 4096)
        if data_char.ndim == 3:  # (N, 33, 4096)
            data_char = data_char[:, None, :, :]
        if data_none_char.ndim == 3:  # (N, 33, 4096)
            data_none_char = data_none_char[:, None, :, :]
        
        print("data_char: ", data_char.shape)
        print("data_none_char: ", data_none_char.shape)

        # Check if JSON file exists
        if not os.path.exists(json_filepath):
            print(f"JSON file not found: {json_filepath}")
            continue

        # Load inconsistent indices from JSON
        with open(json_filepath, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        inconsistent_indices = []
        for idx, entry in enumerate(data.get("data", [])):
            # Keys based on role names: "{task} expert" -> "answer_{task}_expert"
            #                           "non {task} expert" -> "answer_non_{task}_expert"
            task_key = task.replace("_", " ")  # e.g., "college_math" -> "college math"
            expert_key = f"answer_{task_key}_expert".replace(" ", "_")  # "answer_college_math_expert"
            non_expert_key = f"answer_non_{task_key}_expert".replace(" ", "_")  # "answer_non_college_math_expert"

            ans_expert = entry.get(expert_key)
            ans_non_expert = entry.get(non_expert_key)
            if ans_expert != ans_non_expert:
                inconsistent_indices.append(idx)

        if not inconsistent_indices:
            print(f"No inconsistent samples found for task: {task}")
            continue

        
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
    char_mean_filepath = os.path.join(save_path, f"diff_mean_{size}.npy")
    np.save(char_mean_filepath, char_mean)
    print(f"All char mean saved to {char_mean_filepath}")
else:
    print("No char differences found across tasks.")

if all_none_char_diff_data:
    combined_none_char_diff = np.concatenate(all_none_char_diff_data, axis=0)  # Combine along sample axis
    none_char_mean = combined_none_char_diff.mean(axis=0, keepdims=True)  # Compute mean across all samples
    none_char_mean_filepath = os.path.join(save_path, f"none_diff_mean_{size}.npy")
    np.save(none_char_mean_filepath, none_char_mean)
    print(f"None-char mean saved to {none_char_mean_filepath}")
else:
    print("No none-char differences found across tasks.")
