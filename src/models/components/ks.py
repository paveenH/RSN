#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:42:23 2025

@author: paveenhuang

This script computes the KS test (Kolmogorov-Smirnov) for each neuron 
based on inconsistent samples between Expert and None-Expert.
Neurons with p-value < 0.05 are marked as "role-sensitive".
"""

import os
import numpy as np
import json
from scipy.stats import ks_2samp
from tqdm import tqdm

model = "llama3"
size = "8B"

# Path setup
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

# Initialize lists to collect hidden states from inconsistent samples across tasks
all_expert_hidden_states = []
all_none_expert_hidden_states = []

for task in TASKS:
    # Construct file paths for hidden states
    expert_file = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
    none_expert_file = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")

    # Construct file path for JSON answers
    json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")

    # Check if NPY and JSON files exist
    if not os.path.exists(expert_file):
        print(f"Expert hidden states not found for task: {task}, skipping.")
        continue
    if not os.path.exists(none_expert_file):
        print(f"Non-expert hidden states not found for task: {task}, skipping.")
        continue
    if not os.path.exists(json_filepath):
        print(f"JSON file not found for task: {task}, skipping.")
        continue

    # Load the hidden states
    print(f"Loading hidden states for task: {task}")
    expert_data = np.load(expert_file)  # Shape: (num_expert_samples, 1, num_layers, hidden_size)
    none_expert_data = np.load(none_expert_file)  # Shape: (num_none_expert_samples, 1, num_layers, hidden_size)

    # Remove the time dimension
    expert_data = expert_data.squeeze(axis=1)  # Shape: (num_expert_samples, num_layers, hidden_size)
    none_expert_data = none_expert_data.squeeze(axis=1)  # Shape: (num_none_expert_samples, num_layers, hidden_size)

    # Load JSON to identify inconsistent samples
    with open(json_filepath, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Find indices where Expert != None-Expert answers
    inconsistent_indices = []
    for idx, entry in enumerate(data.get("data", [])):
        ans_none = entry.get(f"answer_none_{task}")
        ans_expert = entry.get(f"answer_{task}")
        if ans_none != ans_expert:
            inconsistent_indices.append(idx)

    if not inconsistent_indices:
        print(f"No inconsistent samples found for task: {task}, skipping.")
        continue

    # Extract inconsistent samples
    # Ensure indices are valid for both expert_data and none_expert_data
    if max(inconsistent_indices) >= expert_data.shape[0] or max(inconsistent_indices) >= none_expert_data.shape[0]:
        print(f"Warning: Some inconsistent index out of range for task {task}, skipping those samples.")
        # Filter out invalid indices
        valid_indices = [i for i in inconsistent_indices if i < expert_data.shape[0] and i < none_expert_data.shape[0]]
        if not valid_indices:
            continue
        expert_data_diff = expert_data[valid_indices, ...]
        none_expert_data_diff = none_expert_data[valid_indices, ...]
    else:
        expert_data_diff = expert_data[inconsistent_indices, ...]
        none_expert_data_diff = none_expert_data[inconsistent_indices, ...]

    print(f"Task {task}: total samples = {expert_data.shape[0]}, inconsistent samples used = {expert_data_diff.shape[0]}")

    # Convert to float64 and clip to avoid overflow
    expert_data_diff = expert_data_diff.astype(np.float64)
    none_expert_data_diff = none_expert_data_diff.astype(np.float64)
    expert_data_diff = np.clip(expert_data_diff, -1e6, 1e6)
    none_expert_data_diff = np.clip(none_expert_data_diff, -1e6, 1e6)

    # Append to overall lists
    all_expert_hidden_states.append(expert_data_diff)
    all_none_expert_hidden_states.append(none_expert_data_diff)

# After processing all tasks, combine the inconsistent-sample data
if not all_expert_hidden_states or not all_none_expert_hidden_states:
    raise ValueError("No valid inconsistent samples found across all tasks.")

expert_hidden_states = np.concatenate(all_expert_hidden_states, axis=0)  # Shape: (total_inconsist_samples, num_layers, hidden_size)
none_expert_hidden_states = np.concatenate(all_none_expert_hidden_states, axis=0)  # Same shape

num_expert_samples, num_layers, hidden_size = expert_hidden_states.shape
num_none_expert_samples, num_layers_none, hidden_size_none = none_expert_hidden_states.shape

assert (num_expert_samples == num_none_expert_samples), "Mismatched inconsistent sample counts."
assert (num_layers == num_layers_none and hidden_size == hidden_size_none), "Shape mismatch in layers/hidden_size."

print(f"Total inconsistent samples: {num_expert_samples}")
print(f"Number of layers: {num_layers}")
print(f"Hidden size per layer: {hidden_size}")

# ================= KS Test =====================
# Initialize arrays to store KS statistics and p-values
ks_statistics = np.zeros(num_layers, dtype=np.float64)
p_values = np.ones(num_layers, dtype=np.float64)

for layer in tqdm(range(num_layers), desc="KS Test per Layer"):
    # Flatten all activations in this layer across all samples
    expert_activations = expert_hidden_states[:, layer, :].flatten()
    none_expert_activations = none_expert_hidden_states[:, layer, :].flatten()

    # Perform KS test between the two distributions
    statistic, p_value = ks_2samp(expert_activations, none_expert_activations)
    ks_statistics[layer] = statistic
    p_values[layer] = p_value

# Save results
save_path = os.path.join(current_path, "ks_test_results_layerwise", model)
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, f"ks_layerwise_statistics_{size}.npy"), ks_statistics)
np.save(os.path.join(save_path, f"ks_layerwise_p_values_{size}.npy"), p_values)

print(f"Saved KS statistics to: {save_path}")

