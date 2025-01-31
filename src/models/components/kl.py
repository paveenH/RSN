#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:29:48 2025

@author: paveenhuang
"""

import os
import numpy as np
import json
import argparse
from scipy.stats import entropy
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Compute KL Divergence for neurons between expert and non-expert (inconsistent samples only)")
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
parser.add_argument("top_percentage", type=float, default=0.5, help="Top percentage of neurons to select based on KL divergence")
args = parser.parse_args()

model = args.model
size = args.size
top_percentage = args.top_percentage

# # Fixed parameters
# model = "llama3"
# size = "3B"
# top_percentage = 0.5

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

print(f"Total inconsistent samples for KL: {num_expert_samples}")
print(f"Number of layers: {num_layers}")
print(f"Hidden size per layer: {hidden_size}")

# ============ Compute KL Divergence ============

# 1. Determine global bin range
all_data = np.concatenate([expert_hidden_states, none_expert_hidden_states], axis=0)
global_min = all_data.min()
global_max = all_data.max()

# 2. Decide num_bins
# Use sqrt(N) with a max cap of 100
num_bins = min(int(np.ceil(np.sqrt(num_expert_samples))), 100)
print(f"Using {num_bins} bins for histogram.")

bins = np.linspace(global_min, global_max, num_bins + 1)

# 3. Initialize KL matrix
kl_divergences = np.zeros((num_layers, hidden_size), dtype=np.float64)

print("Computing KL Divergence for each neuron...")
for layer in tqdm(range(num_layers), desc="Layers"):
    for neuron in range(hidden_size):
        # Extract activations for the current neuron
        expert_activations = expert_hidden_states[:, layer, neuron]
        none_expert_activations = none_expert_hidden_states[:, layer, neuron]

        # Compute histograms
        expert_hist, _ = np.histogram(expert_activations, bins=bins, density=True)
        none_expert_hist, _ = np.histogram(none_expert_activations, bins=bins, density=True)

        # Avoid zero
        epsilon = 1e-10
        expert_hist += epsilon
        none_expert_hist += epsilon

        # Normalize
        expert_hist /= expert_hist.sum()
        none_expert_hist /= none_expert_hist.sum()

        # KL(expert || none_expert)
        kl_value = entropy(expert_hist, none_expert_hist)
        kl_divergences[layer, neuron] = kl_value

# 4. Flatten and sort
kl_flat = kl_divergences.flatten()
num_neurons = kl_flat.shape[0]
top_k = int(np.ceil((top_percentage / 100) * num_neurons))
top_k = max(top_k, 1)

top_indices = np.argsort(kl_flat)[-top_k:]  # top_k largest

# Convert flat indices -> (layer, neuron)
top_neurons = [(int(idx // hidden_size), int(idx % hidden_size)) for idx in top_indices]

print(f"Top {top_percentage}% neurons based on KL Divergence (inconsistent samples only):")
for layer, neuron in top_neurons:
    print(f"Layer {layer}, Neuron {neuron}, KL Divergence: {kl_divergences[layer, neuron]:.6f}")

# Save results
save_path = os.path.join(current_path, "kl_divergence_results", model)
os.makedirs(save_path, exist_ok=True)

kl_save_path = os.path.join(save_path, f"kl_divergence_inconsistent_{size}.npy")
np.save(kl_save_path, kl_divergences)
print(f"KL Divergence matrix saved to {kl_save_path}")

top_neurons_save_path = os.path.join(save_path, f"top_{top_percentage}_percent_neurons_inconsistent_{size}.json")
with open(top_neurons_save_path, 'w', encoding='utf-8') as f:
    json.dump(top_neurons, f, ensure_ascii=False, indent=4)

print(f"Top neurons saved to {top_neurons_save_path}")