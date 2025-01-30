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

parser = argparse.ArgumentParser(description="Compute KL Divergence for neurons between expert and non-expert")
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
parser.add_argument("--num_bins", type=int, default=100, help="Number of bins for histogram")
parser.add_argument("--top_percentage", type=float, default=0.5, help="Top percentage of neurons to select based on KL divergence")
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

# Define the task list
TASKS = ["sociology", "virology"]

# Initialize lists to collect hidden states
all_expert_hidden_states = []
all_none_expert_hidden_states = []

# Load hidden states for each task
for task in TASKS:
    expert_file = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
    none_expert_file = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")

    if not os.path.exists(expert_file):
        print(f"Expert hidden states not found for task: {task}, skipping.")
        continue
    if not os.path.exists(none_expert_file):
        print(f"Non-expert hidden states not found for task: {task}, skipping.")
        continue

    print(f"Loading hidden states for task: {task}")
    expert_data = np.load(expert_file)  # Shape: (num_expert_samples, 1, num_layers, hidden_size)
    none_expert_data = np.load(none_expert_file)  # Shape: (num_none_expert_samples, 1, num_layers, hidden_size)

    # Remove the time dimension (since it's always 1)
    expert_data = expert_data.squeeze(axis=1)  # Shape: (num_expert_samples, num_layers, hidden_size)
    none_expert_data = none_expert_data.squeeze(axis=1)  # Shape: (num_none_expert_samples, num_layers, hidden_size)

    all_expert_hidden_states.append(expert_data)
    all_none_expert_hidden_states.append(none_expert_data)

# Ensure at least some data was loaded
if not all_expert_hidden_states or not all_none_expert_hidden_states:
    raise ValueError("No valid expert or non-expert hidden states found across all tasks.")

# Concatenate all loaded data across tasks
expert_hidden_states = np.concatenate(all_expert_hidden_states, axis=0)  # Combine along sample axis
none_expert_hidden_states = np.concatenate(all_none_expert_hidden_states, axis=0)  # Combine along sample axis

# Get the final shape
num_expert_samples, num_layers, hidden_size = expert_hidden_states.shape
num_none_expert_samples, num_layers_none, hidden_size_none = none_expert_hidden_states.shape

# Ensure dimensions match
assert num_layers == num_layers_none and hidden_size == hidden_size_none, "Expert and Non-expert hidden states shape mismatch."

print(f"Number of expert samples: {num_expert_samples}")
print(f"Number of non-expert samples: {num_none_expert_samples}")
print(f"Number of layers: {num_layers}")
print(f"Hidden size per layer: {hidden_size}")

# Compute global bin range for KL divergence calculation
all_data = np.concatenate([expert_hidden_states, none_expert_hidden_states], axis=0)
global_min = all_data.min()
global_max = all_data.max()

if num_expert_samples < 500:
    # Sturges’ Rule
    num_bins = int(np.ceil(np.log2(num_expert_samples) + 1))
elif num_expert_samples > 1000:
    # Scott’s Rule
    all_activations = np.concatenate([expert_hidden_states.flatten(), none_expert_hidden_states.flatten()])
    std_dev = np.std(all_activations)
    num_bins = int(np.ceil((3.49 * std_dev) / (num_expert_samples ** (1/3))))
else:
    # 500 ≤ N ≤ 1000
    sturges_bins = int(np.ceil(np.log2(num_expert_samples) + 1))
    all_activations = np.concatenate([expert_hidden_states.flatten(), none_expert_hidden_states.flatten()])
    std_dev = np.std(all_activations)
    scott_bins = int(np.ceil((3.49 * std_dev) / (num_expert_samples ** (1/3))))
    
    # mean between Sturges and Scott
    num_bins = int(np.ceil((sturges_bins + scott_bins) / 2))

# bins
bins = np.linspace(global_min, global_max, num_bins + 1)

# Initialize KL divergence storage
kl_divergences = np.zeros((num_layers, hidden_size))

print("Computing KL Divergence for each neuron...")
# Compute KL divergence per neuron
for layer in tqdm(range(num_layers), desc="Layers"):
    for neuron in range(hidden_size):
        # Extract activations for the current neuron
        expert_activations = expert_hidden_states[:, layer, neuron]
        none_expert_activations = none_expert_hidden_states[:, layer, neuron]

        # Compute histograms
        expert_hist, _ = np.histogram(expert_activations, bins=bins, density=True)
        none_expert_hist, _ = np.histogram(none_expert_activations, bins=bins, density=True)

        # Add a small epsilon to avoid zero values
        epsilon = 1e-10
        expert_hist += epsilon
        none_expert_hist += epsilon

        # Normalize the histograms
        expert_hist /= expert_hist.sum()
        none_expert_hist /= none_expert_hist.sum()

        # Compute KL divergence: KL(expert || none_expert)
        kl = entropy(expert_hist, none_expert_hist)
        kl_divergences[layer, neuron] = kl

# Flatten KL divergence values and sort
kl_flat = kl_divergences.flatten()
num_neurons = kl_flat.shape[0]
top_k = int(np.ceil((top_percentage / 100) * num_neurons))
top_k = max(top_k, 1)  # Ensure at least one neuron is selected

# Get the indices of top_k neurons
top_indices = np.argsort(kl_flat)[-top_k:]

# Convert the flat indices back to (layer, neuron) format and cast to Python int
top_neurons = [(int(idx // hidden_size), int(idx % hidden_size)) for idx in top_indices]

print(f"Top {top_percentage}% neurons based on KL Divergence:")
for layer, neuron in top_neurons:
    print(f"Layer {layer}, Neuron {neuron}, KL Divergence: {kl_divergences[layer, neuron]:.6f}")

# Save results
save_path = os.path.join(current_path, "kl_divergence_results", model)
os.makedirs(save_path, exist_ok=True)
kl_save_path = os.path.join(save_path, f"kl_divergence_{size}.npy")
np.save(kl_save_path, kl_divergences)
print(f"KL Divergence matrix saved to {kl_save_path}")

# Save top neurons as JSON (Fixing TypeError)
top_neurons_save_path = os.path.join(save_path, f"top_{top_percentage}_percent_neurons_{size}.json")
with open(top_neurons_save_path, 'w', encoding='utf-8') as f:
    json.dump(top_neurons, f, ensure_ascii=False, indent=4)

print(f"Top neurons saved to {top_neurons_save_path}")