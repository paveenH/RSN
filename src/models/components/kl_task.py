#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:43:46 2025

Modified to compute per-task KL Divergence for neurons between expert and non-expert
(using only inconsistent samples). For each task the KL divergence matrix has shape:
    (num_layers, hidden_size)

All task results (expected 57 tasks) are stacked into a final matrix of shape:
    (num_tasks, num_layers, hidden_size)

The KL divergence is computed for each neuron as:
    KL(expert || non_expert)
    
@author: paveenhuang
"""


import os
import numpy as np
import json
import argparse
from scipy.stats import entropy

import sys

if os.fork():
    sys.exit()  


# ------------------------------
# Parse arguments
# ------------------------------
parser = argparse.ArgumentParser(
    description="Compute per-task KL Divergence for neurons between expert and non-expert (inconsistent samples only)"
)
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
args = parser.parse_args()

model = args.model
size = args.size

# model = "llama3"
# size = "3B"

# ------------------------------
# Path setup
# ------------------------------
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
json_path = os.path.join(current_path, "answer", model)
save_path = os.path.join(current_path, "dice_kl", model)
os.makedirs(save_path, exist_ok=True)

# ------------------------------
# Task list (expected 57 tasks)
# ------------------------------
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

# To save KL divergence result in each task
kl_results = {}

# ------------------------------
# Calculate KL divergence for each task
# ------------------------------
for task in TASKS:
    print(f"\nProcessing task: {task}")
    
    # Path
    expert_file = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
    none_expert_file = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")
    json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")
    
    # Check file 
    if not os.path.exists(expert_file):
        print(f"Expert hidden states file not found for task: {task}, skipping this task.")
        continue
    if not os.path.exists(none_expert_file):
        print(f"Non-expert hidden states file not found for task: {task}, skipping this task.")
        continue
    if not os.path.exists(json_filepath):
        print(f"JSON file not found for task: {task}, skipping this task.")
        continue
    
    # Load hidden states (num_samples, 1, num_layers, hidden_size)
    try:
        expert_data = np.load(expert_file)
        none_expert_data = np.load(none_expert_file)
    except Exception as e:
        print(f"Error loading hidden states for task {task}: {e}")
        continue
    
    # Squeeze time
    if expert_data.ndim == 4:
        expert_data = np.squeeze(expert_data, axis=1)
    if none_expert_data.ndim == 4:
        none_expert_data = np.squeeze(none_expert_data, axis=1)
    
    # Load JSON 
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON for task {task}: {e}")
        continue
    
    inconsistent_indices = []
    for idx, entry in enumerate(json_data.get("data", [])):
        ans_expert = entry.get(f"answer_{task}")
        ans_none = entry.get(f"answer_none_{task}")
        if ans_expert != ans_none:
            inconsistent_indices.append(idx)
    
    # If there are no inconsistent samples, fill them with NaN matrix (make sure the result matrix still has entries for this task)
    if not inconsistent_indices:
        print(f"No inconsistent samples found for task: {task}. Filling with NaNs.")
        num_layers = expert_data.shape[1]
        hidden_size = expert_data.shape[2]
        kl_results[task] = np.full((num_layers, hidden_size), np.nan)
        continue

    # Check the index range and extract valid inconsistent samples
    if max(inconsistent_indices) >= expert_data.shape[0] or max(inconsistent_indices) >= none_expert_data.shape[0]:
        valid_indices = [i for i in inconsistent_indices if i < expert_data.shape[0] and i < none_expert_data.shape[0]]
        if not valid_indices:
            print(f"No valid inconsistent sample indices for task: {task}. Filling with NaNs.")
            num_layers = expert_data.shape[1]
            hidden_size = expert_data.shape[2]
            kl_results[task] = np.full((num_layers, hidden_size), np.nan)
            continue
        expert_data_diff = expert_data[valid_indices, ...]
        none_expert_data_diff = none_expert_data[valid_indices, ...]
    else:
        expert_data_diff = expert_data[inconsistent_indices, ...]
        none_expert_data_diff = none_expert_data[inconsistent_indices, ...]
    
    print(f"Task {task}: total samples = {expert_data.shape[0]}, inconsistent samples used = {expert_data_diff.shape[0]}")
    
    # Convert the data to float64 and clip it to prevent the value from being too large or too small.
    expert_data_diff = expert_data_diff.astype(np.float64)
    none_expert_data_diff = none_expert_data_diff.astype(np.float64)
    expert_data_diff = np.clip(expert_data_diff, -1e6, 1e6)
    none_expert_data_diff = np.clip(none_expert_data_diff, -1e6, 1e6)
    
    # Determine the global bin range for the current task (based on two types of data)
    combined_data = np.concatenate([expert_data_diff, none_expert_data_diff], axis=0)
    global_min = combined_data.min()
    global_max = combined_data.max()
    
    # Histogram bins based on the number of inconsistent samples
    n_samples = expert_data_diff.shape[0]
    num_bins = min(int(np.ceil(np.sqrt(n_samples))), 100)
    bins = np.linspace(global_min, global_max, num_bins + 1)
    
    num_layers = expert_data_diff.shape[1]
    hidden_size = expert_data_diff.shape[2]
    
    # Initialize the KL divergence matrix of the current task
    kl_task = np.zeros((num_layers, hidden_size), dtype=np.float64)
    
    # Calculate the KL divergence for each neuron (each unit in each layer)
    for layer in range(num_layers):
        for neuron in range(hidden_size):
            expert_activations = expert_data_diff[:, layer, neuron]
            none_expert_activations = none_expert_data_diff[:, layer, neuron]
            
            expert_hist, _ = np.histogram(expert_activations, bins=bins, density=True)
            none_expert_hist, _ = np.histogram(none_expert_activations, bins=bins, density=True)
            
            # To prevent zero values ​​from appearing in the histogram, add a small epsilon
            epsilon = 1e-10
            expert_hist += epsilon
            none_expert_hist += epsilon
            
            # Normalized Histogram
            expert_hist /= expert_hist.sum()
            none_expert_hist /= none_expert_hist.sum()
            
            # KL Divergence: KL(expert || none_expert)
            kl_value = entropy(expert_hist, none_expert_hist)
            kl_task[layer, neuron] = kl_value
            
    # Save
    kl_results[task] = kl_task

# ------------------------------
# Stack the results of all tasks into the final matrix in the order of the TASKS list
# ------------------------------
final_kl_matrix_list = []
for task in TASKS:
    if task in kl_results:
        final_kl_matrix_list.append(kl_results[task])
    else:
        print(f"Task {task} missing in results, filling with NaNs.")
        sample_shape = list(kl_results.values())[0].shape if len(kl_results) > 0 else (0, 0)
        final_kl_matrix_list.append(np.full(sample_shape, np.nan))

final_kl_matrix = np.array(final_kl_matrix_list)
print(f"\nFinal KL divergence matrix shape: {final_kl_matrix.shape}")

# ------------------------------
# Save
# ------------------------------
# kl_save_path = os.path.join(save_path, f"kl_divergence_inconsistent_per_task_{size}.npy")
# np.save(kl_save_path, final_kl_matrix)
# print(f"KL Divergence per task matrix saved to {kl_save_path}")

# ----------------------------------------------------------------------------
# Dice coefficient
# ----------------------------------------------------------------------------

def dice_coefficient(A, B):
    """
    Compute the Dice (Sørensen–Dice) coefficient between two binary masks A and B.
    A and B should be numpy arrays containing 0s and 1s.
    """
    intersection = np.sum(A * B)
    sum_A = np.sum(A)
    sum_B = np.sum(B)
    # If both masks are empty, they are considered identical.
    if sum_A + sum_B == 0:
        return 1.0
    return 2 * intersection / (sum_A + sum_B)

# Generate a binarization mask for each task:
# For each layer, select the top (hidden_size // 200) neuron positions 
# (sorted based on |value_diff|) and mark them as 1, and the rest as 0
num_tasks, num_layers, hidden_size = final_kl_matrix.shape
masks = []

for t in range(num_tasks):
    mask_task = np.zeros((num_layers, hidden_size))
    for layer in range(1, num_layers): # exdlude 
        # Calculate the absolute value of value_diff in the current layer
        layer_values = np.abs(final_kl_matrix[t, layer, :])
        top_n = max(hidden_size // 200, 1)
        # Select top_n neuron indices (based on absolute magnitude)
        top_indices = np.argsort(layer_values)[-top_n:]
        mask_layer = np.zeros(hidden_size)
        mask_layer[top_indices] = 1
        mask_task[layer, :] = mask_layer
    masks.append(mask_task)

masks = np.array(masks)  # shape: (num_tasks, num_layers, hidden_size)

# Compute the Dice coefficient matrix between tasks (shape: num_tasks x num_tasks)
dice_matrix = np.zeros((num_tasks, num_tasks))
for i in range(num_tasks):
    for j in range(num_tasks):
        # Dice coefficient is calculated after flattening the mask of each task
        dice_matrix[i, j] = dice_coefficient(masks[i].flatten(), masks[j].flatten())

print(f"Dice similarity matrix shape: {dice_matrix.shape}")

# Save the Dice similarity matrix
dice_save_path = os.path.join(save_path, f"dice_kl_{size}.npy")
np.save(dice_save_path, dice_matrix)
print(f"Dice similarity matrix saved to: {dice_save_path}")