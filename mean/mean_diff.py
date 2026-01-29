#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:54:17 2024
@description:
    Mean NMD for Expert - NoExpert in diff answer-pairs
    Uses incremental mean calculation to avoid memory issues.

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
AnswerName = f"answer_{TYPE}_logits"  # "answer_non_logits" or "answer_non"

# Base directory (NCHC or local)
DIR = "/work/d12922004/RolePlaying/components"
# DIR = "/data2/paveen/RolePlaying/components"  # Local alternative

# Paths
json_path = os.path.join(DIR, AnswerName, model)
hidden_states_path = os.path.join(DIR, f"hidden_states_{TYPE}", model)

if "logits" in AnswerName:
    save_path = os.path.join(DIR, "hidden_states_mean", f"{model}_{TYPE}_logits")
else:
    save_path = os.path.join(DIR, "hidden_states_mean", f"{model}_{TYPE}")
os.makedirs(save_path, exist_ok=True)
print("save path: ", save_path)

# Use incremental mean calculation to avoid memory issues
# mean = sum / count, computed incrementally as: new_mean = old_mean + (new_data - old_mean) / new_count
char_sum = None
none_char_sum = None
char_count = 0
none_char_count = 0

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

        # Check if JSON file exists first (before loading large npy files)
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

        # Load npy files with memory mapping to reduce memory usage
        data_char = np.load(data_char_filepath, mmap_mode='r')
        data_none_char = np.load(data_none_char_filepath, mmap_mode='r')

        # Get only inconsistent samples and compute their sum
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(inconsistent_indices), batch_size):
            batch_indices = inconsistent_indices[i:i+batch_size]

            # Load batch data
            char_batch = data_char[batch_indices, ...]
            none_char_batch = data_none_char[batch_indices, ...]

            # Ensure correct shape (N, 1, layers, hidden)
            if char_batch.ndim == 3:
                char_batch = char_batch[:, None, :, :]
            if none_char_batch.ndim == 3:
                none_char_batch = none_char_batch[:, None, :, :]

            # Convert to float32 for accumulation to avoid overflow
            char_batch = char_batch.astype(np.float32)
            none_char_batch = none_char_batch.astype(np.float32)

            # Accumulate sum
            batch_char_sum = char_batch.sum(axis=0)
            batch_none_char_sum = none_char_batch.sum(axis=0)

            if char_sum is None:
                char_sum = batch_char_sum
                none_char_sum = batch_none_char_sum
            else:
                char_sum += batch_char_sum
                none_char_sum += batch_none_char_sum

            char_count += len(batch_indices)
            none_char_count += len(batch_indices)

        print(f"Processed task: {task}, inconsistent samples: {len(inconsistent_indices)}, total so far: {char_count}")

    except Exception as e:
        print(f"Error processing task {task}: {e}")

# Compute final means
if char_count > 0:
    char_mean = (char_sum / char_count).astype(np.float16)  # Convert back to float16 for storage
    char_mean = char_mean[None, ...]  # Add batch dimension: (1, 1, layers, hidden)
    char_mean_filepath = os.path.join(save_path, f"diff_mean_{size}.npy")
    np.save(char_mean_filepath, char_mean)
    print(f"Expert mean saved to {char_mean_filepath}, shape: {char_mean.shape}, total samples: {char_count}")
else:
    print("No expert differences found across tasks.")

if none_char_count > 0:
    none_char_mean = (none_char_sum / none_char_count).astype(np.float16)
    none_char_mean = none_char_mean[None, ...]
    none_char_mean_filepath = os.path.join(save_path, f"none_diff_mean_{size}.npy")
    np.save(none_char_mean_filepath, none_char_mean)
    print(f"Non-expert mean saved to {none_char_mean_filepath}, shape: {none_char_mean.shape}, total samples: {none_char_count}")
else:
    print("No non-expert differences found across tasks.")
