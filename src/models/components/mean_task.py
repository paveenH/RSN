#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:03:41 2025

Script to compute the difference between the mean hidden states of char and none‑char
(from inconsistent samples only) for each task, and save the result.

The difference is computed as:
    value_diff = char_mean - none_char_mean

The final saved array has shape: (num_tasks, num_layers, hidden_size).

Additionally, this script computes a binary mask for each task that selects the top neurons
in each layer (top neurons = top (hidden_size // 200) based on absolute value of value_diff).
Then, the Dice coefficient (Sørensen–Dice Coefficient) is computed for each pair of tasks,
yielding a similarity matrix of shape (num_tasks, num_tasks) which is saved.

@author: paveenhuang
"""

import os
import json
import argparse
import numpy as np

# -------------------------------
# Parse command-line arguments
# -------------------------------
parser = argparse.ArgumentParser(
    description="Compute difference between char and none‑char mean hidden states (inconsistent samples only)"
)
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
args = parser.parse_args()

model = args.model
size = args.size

# # Fixed parameters
# model = "llama3"
# size = "3B"
# top_percentage = 0.5

# -------------------------------
# Path definition
# -------------------------------
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
json_path = os.path.join(current_path, "answer", model)
save_path = os.path.join(current_path, "diff_results", model)
os.makedirs(save_path, exist_ok=True)

# -------------------------------
# Task list (e.g., 57 tasks)
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

# Dictionary to store value_diff for each task
value_diff_tasks = {}

# -------------------------------
# Process each task to compute value_diff
# -------------------------------
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

    # Shape: (num_samples, 1, num_layers, hidden_size); squeeze out time dimension if present
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
    
    # Find indices of inconsistent answers
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
    
    # Convert data types and clip to avoid extreme values
    char_data_inconsistent = char_data_inconsistent.astype(np.float64)
    none_char_data_inconsistent = none_char_data_inconsistent.astype(np.float64)
    char_data_inconsistent = np.clip(char_data_inconsistent, -1e6, 1e6)
    none_char_data_inconsistent = np.clip(none_char_data_inconsistent, -1e6, 1e6)
    
    # Calculate the mean of inconsistent samples for each task 
    # Resulting shape: (num_layers, hidden_size)
    char_mean = char_data_inconsistent.mean(axis=0)
    none_char_mean = none_char_data_inconsistent.mean(axis=0)
    
    # Compute value difference: (char_mean - none_char_mean)
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

# # Save the value_diff array
# value_diff_save_path = os.path.join(save_path, f"value_diff_inconsistent_{size}.npy")
# np.save(value_diff_save_path, value_diff_array)
# print(f"Value difference per task saved to: {value_diff_save_path}")

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
num_tasks, num_layers, hidden_size = value_diff_array.shape
masks = []

for t in range(num_tasks):
    mask_task = np.zeros((num_layers, hidden_size))
    for layer in range(1, num_layers): # exdlude 
        # Calculate the absolute value of value_diff in the current layer
        layer_values = np.abs(value_diff_array[t, layer, :])
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
dice_save_path = os.path.join(save_path, f"dice_similarity_matrix_inconsistent_{size}.npy")
np.save(dice_save_path, dice_matrix)
print(f"Dice similarity matrix saved to: {dice_save_path}")

# ----------------------------------------------------------------------------
# Random mask
# ----------------------------------------------------------------------------

random_masks = []
for t in range(num_tasks):
    mask_task = np.zeros((num_layers, hidden_size))
    for layer in range(1, num_layers):  # 排除 embedding 层
        top_n = max(hidden_size // 200, 1)
        random_indices = np.random.choice(hidden_size, size=top_n, replace=False)
        mask_layer = np.zeros(hidden_size)
        mask_layer[random_indices] = 1
        mask_task[layer, :] = mask_layer
    random_masks.append(mask_task)

random_masks = np.array(random_masks) 

random_dice_matrix = np.zeros((num_tasks, num_tasks))
for i in range(num_tasks):
    for j in range(num_tasks):
        random_dice_matrix[i, j] = dice_coefficient(random_masks[i].flatten(), random_masks[j].flatten())

random_mean_value = random_dice_matrix.mean()
print(f"Mean Dice similarity (Random): {random_mean_value:.4f}")

random_dice_save_path = os.path.join(save_path, f"random_dice_similarity_matrix_{size}.npy")
np.save(random_dice_save_path, random_dice_matrix)
print(f"Random Dice similarity matrix saved to: {random_dice_save_path}")