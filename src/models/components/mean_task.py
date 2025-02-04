#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:03:41 2025

Script to compute the difference between the mean hidden states of char and none‑char
(from inconsistent samples only) for each task, and save the result.

The difference is computed as:
    value_diff = char_mean - none_char_mean

The final saved array has shape: (num_tasks, num_layers, hidden_size).

@author: paveenhuang
"""

import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Compute KL Divergence for neurons between expert and non-expert (inconsistent samples only)")
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
args = parser.parse_args()

model = args.model
size = args.size

# # Fixed parameters
# model = "llama3"
# size = "3B"

# Path definition
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
json_path = os.path.join(current_path, "answer", model)

# Task list
TASKS = [
    "abstract_algebra", "anatomy", "astronomy", 
    "business_ethics", "clinical_knowledge", 
    "college_biology", "college_chemistry", "college_computer_science", "college_medicine", 
    "college_mathematics", "college_physics", "computer_security", "conceptual_physics", 
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", 
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", 
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics", 
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", 
    "high_school_physics", "high_school_psychology", "high_school_statistics", 
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", 
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", 
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", 
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", 
    "professional_medicine", "professional_psychology", "public_relations", "security_studies", 
    "sociology", "us_foreign_policy", "virology", "world_religions"
]

# char_mean - none_char_mean
value_diff_tasks = {}

for task in TASKS:
    print(f"Processing task: {task}")
    
    char_filepath = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
    none_char_filepath = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")
    json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")
    
    if not os.path.exists(char_filepath):
        print(f"Char hidden states file not found for task: {task}, skipping.")
        continue
    if not os.path.exists(none_char_filepath):
        print(f"None‑char hidden states file not found for task: {task}, skipping.")
        continue
    if not os.path.exists(json_filepath):
        print(f"JSON file not found for task: {task}, skipping.")
        continue
    
    try:
        char_data = np.load(char_filepath)
        none_char_data = np.load(none_char_filepath)
    except Exception as e:
        print(f"Error loading hidden states for task {task}: {e}")
        continue

    # Shape: (num_samples, 1, num_layers, hidden_size), squeeze time 
    if char_data.ndim == 4:
        char_data = np.squeeze(char_data, axis=1)
    if none_char_data.ndim == 4:
        none_char_data = np.squeeze(none_char_data, axis=1)
    
    # Load JSON
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON for task {task}: {e}")
        continue
    
    # Find inconsistent answers
    inconsistent_indices = []
    for idx, entry in enumerate(json_data.get("data", [])):
        answer_char = entry.get(f"answer_{task}")
        answer_none = entry.get(f"answer_none_{task}")
        if answer_char != answer_none:
            inconsistent_indices.append(idx)
    
    if not inconsistent_indices:
        print(f"No inconsistent samples found for task: {task}, skipping.")
        continue
    
    # Extract inconsistent hidden states
    try:
        char_data_inconsistent = char_data[inconsistent_indices, ...]
        none_char_data_inconsistent = none_char_data[inconsistent_indices, ...]
    except Exception as e:
        print(f"Error extracting inconsistent samples for task {task}: {e}")
        continue
    
    print(f"Task {task}: total samples = {char_data.shape[0]}, inconsistent samples = {len(inconsistent_indices)}")
    
    # Convert data types and clip to avoid extreme value interference
    char_data_inconsistent = char_data_inconsistent.astype(np.float64)
    none_char_data_inconsistent = none_char_data_inconsistent.astype(np.float64)
    char_data_inconsistent = np.clip(char_data_inconsistent, -1e6, 1e6)
    none_char_data_inconsistent = np.clip(none_char_data_inconsistent, -1e6, 1e6)
    
    # Calculate the mean of all inconsistent samples for each task 
    # (axis=0, resulting in a shape of (num_layers, hidden_size))
    char_mean = char_data_inconsistent.mean(axis=0)
    none_char_mean = none_char_data_inconsistent.mean(axis=0)
    
    # Calculate the difference: (char_mean - none_char_mean)
    value_diff = char_mean - none_char_mean
    
    task_name = task.replace('_', ' ')
    value_diff_tasks[task_name] = value_diff

if not value_diff_tasks:
    raise ValueError("No valid inconsistent samples found across tasks.")

# Sort task names
sorted_tasks = sorted(value_diff_tasks.keys())
print(f"Sorted tasks: {sorted_tasks}")

# Stack the differences for each task into an array of shape (num_tasks, num_layers, hidden_size)
value_diff_array = np.array([value_diff_tasks[task] for task in sorted_tasks])
print(f"Final value diff array shape: {value_diff_array.shape}")

# Save
save_path = os.path.join(current_path, "value_diff_results", model)
os.makedirs(save_path, exist_ok=True)
value_diff_save_path = os.path.join(save_path, f"value_diff_inconsistent_{size}.npy")
np.save(value_diff_save_path, value_diff_array)
print(f"Value difference per task saved to: {value_diff_save_path}")