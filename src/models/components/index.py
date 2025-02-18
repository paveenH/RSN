#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:13:57 2025

Script to count, across all 57 tasks, for each layer, the frequency of neuron indices 
that appear in the top (e.g., top=20) neurons (based on absolute value of the value difference 
between char and none‑char mean hidden states).
The results are saved as a CSV file.
 
The CSV file has columns:
    Layer, Neuron_Index, Count
Additionally, the code prints out, for each layer, the neuron index with the highest count.

@author: paveenhuang
"""

import os
import numpy as np
import json
import csv

# -------------------------------
# Fixed parameters and paths
# -------------------------------
model = "llama3"
size = "3B"
task = "all_mean"  # used to construct file names
# Hidden states and JSON files are assumed to be under:
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
json_path = os.path.join(current_path, "answer", model)
save_dir = os.path.join(current_path, "Index")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# -------------------------------
# TASKS list: 57 tasks
# -------------------------------
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

# Dictionary to store value_diff for each task.
# For each task, value_diff is computed as:
#    value_diff = (char_mean - none_char_mean)
# where char_mean and none_char_mean are the mean hidden states computed over inconsistent samples.
value_diff_tasks = {}

# -------------------------------
# Process each task to compute value_diff
# -------------------------------
for task_name in TASKS:
    print(f"Processing task: {task_name}")
    
    char_filepath = os.path.join(hidden_states_path, f"{task_name}_{task_name}_{size}.npy")
    none_char_filepath = os.path.join(hidden_states_path, f"none_{task_name}_{task_name}_{size}.npy")
    json_filepath = os.path.join(json_path, f"{task_name}_{size}_answers.json")
    
    if not os.path.exists(char_filepath):
        print(f"Char hidden states file not found for task: {task_name}, skipping.")
        continue
    if not os.path.exists(none_char_filepath):
        print(f"None‑char hidden states file not found for task: {task_name}, skipping.")
        continue
    if not os.path.exists(json_filepath):
        print(f"JSON file not found for task: {task_name}, skipping.")
        continue
    
    try:
        char_data = np.load(char_filepath)
        none_char_data = np.load(none_char_filepath)
    except Exception as e:
        print(f"Error loading hidden states for task {task_name}: {e}")
        continue

    # Squeeze out the time dimension if present (assuming shape: (num_samples, 1, num_layers, hidden_size))
    if char_data.ndim == 4:
        char_data = np.squeeze(char_data, axis=1)
    if none_char_data.ndim == 4:
        none_char_data = np.squeeze(none_char_data, axis=1)
    
    # Load JSON file
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON for task {task_name}: {e}")
        continue
    
    # Identify indices of inconsistent samples (where answer_char != answer_none)
    inconsistent_indices = []
    for idx, entry in enumerate(json_data.get("data", [])):
        answer_char = entry.get(f"answer_{task_name}")
        answer_none = entry.get(f"answer_none_{task_name}")
        if answer_char != answer_none:
            inconsistent_indices.append(idx)
    
    if not inconsistent_indices:
        print(f"No inconsistent samples found for task: {task_name}, skipping.")
        continue
    
    try:
        char_data_inconsistent = char_data[inconsistent_indices, ...]
        none_char_data_inconsistent = none_char_data[inconsistent_indices, ...]
    except Exception as e:
        print(f"Error extracting inconsistent samples for task {task_name}: {e}")
        continue
    
    print(f"Task {task_name}: total samples = {char_data.shape[0]}, inconsistent samples = {len(inconsistent_indices)}")
    
    # Convert to float64 and clip extreme values
    char_data_inconsistent = char_data_inconsistent.astype(np.float64)
    none_char_data_inconsistent = none_char_data_inconsistent.astype(np.float64)
    char_data_inconsistent = np.clip(char_data_inconsistent, -1e6, 1e6)
    none_char_data_inconsistent = np.clip(none_char_data_inconsistent, -1e6, 1e6)
    
    # Compute the mean hidden state for each task over inconsistent samples
    char_mean = char_data_inconsistent.mean(axis=0)   # shape: (num_layers, hidden_size)
    none_char_mean = none_char_data_inconsistent.mean(axis=0)
    
    # Compute value difference for the task
    value_diff = char_mean - none_char_mean  # shape: (num_layers, hidden_size)
    
    # Exclude layer 0 (embedding layer)
    value_diff = value_diff[1:, :]  # now shape: (num_layers-1, hidden_size)
    
    # Save in dictionary (convert task name to a more readable format if desired)
    key = task_name.replace('_', ' ')
    value_diff_tasks[key] = value_diff

if not value_diff_tasks:
    raise ValueError("No valid tasks found with inconsistent samples.")

# Sort tasks (for consistency, optional)
sorted_tasks = sorted(value_diff_tasks.keys())
print(f"Sorted tasks: {sorted_tasks}")

# Stack into an array of shape (num_tasks, num_layers, hidden_size)
value_diff_array = np.array([value_diff_tasks[task] for task in sorted_tasks])
print(f"Final value diff array shape: {value_diff_array.shape}")

# -------------------------------
# Count top neurons frequency per layer
# -------------------------------
top_n = 20  # e.g., top=20 neurons per layer
num_tasks, num_layers, hidden_size = value_diff_array.shape

# Initialize a dictionary to hold counts for each layer.
layer_counts = {layer: np.zeros(hidden_size, dtype=int) for layer in range(num_layers)}

for t in range(num_tasks):
    for layer in range(num_layers):
        # For current task and layer, compute absolute values and get indices of top_n neurons.
        layer_values = np.abs(value_diff_array[t, layer, :])
        top_indices = np.argsort(layer_values)[-top_n:]
        # Increase count for these neuron indices.
        layer_counts[layer][top_indices] += 1

# For each layer, determine the most frequently occurring neuron index.
most_important_indices = {}
for layer in range(num_layers):
    idx = np.argmax(layer_counts[layer])
    count = layer_counts[layer][idx]
    most_important_indices[layer] = (idx, count)
    print(f"Layer {layer+1}: Most important neuron index = {idx} (count = {count})")

# -------------------------------
# Save the counts to a CSV file
# -------------------------------
csv_file = os.path.join(save_dir, f"{model}_{size}_top_{top_n}_neuron_frequency.csv")
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    # Write header: Layer, Neuron_Index, Count
    writer.writerow(["Layer", "Neuron_Index", "Count"])
    for layer in range(num_layers):
        for neuron_index in range(hidden_size):
            writer.writerow([layer + 1, neuron_index, layer_counts[layer][neuron_index]])

print(f"Saved top neuron frequency counts to {csv_file}")