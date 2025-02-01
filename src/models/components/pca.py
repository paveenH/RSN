#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:26:59 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:12:52 2025

Author: paveenhuang

Objective: Use PCA to analyze hidden states (from inconsistent samples) and extract the top 0.5% most important neurons based on PCA loadings.
"""

import os
import numpy as np
import json
import argparse
from sklearn.decomposition import PCA

# ------------------ Parameter parsing ------------------
parser = argparse.ArgumentParser(description="Perform PCA on inconsistent samples and extract top neurons based on PCA loadings")
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B or 3B)")
parser.add_argument("top_percentage", type=float, help="Top percentage of neurons to select based on PCA loadings (e.g., 0.5 for top 0.5%)")
args = parser.parse_args()

model = args.model
size = args.size
top_percentage = args.top_percentage

# # Fixed parameters
# model = "llama3"
# size = "3B"
# top_percentage = 0.5

# ------------------ Path definition ------------------
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
json_path = os.path.join(current_path, "answer", model)

# ------------------ Task list ------------------
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

# ------------------ Load inconsistent samples ------------------
all_expert_hidden_states = []
all_none_expert_hidden_states = []

print("Loading inconsistent samples from all tasks...")
for task in TASKS:
    expert_file = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
    none_expert_file = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")
    json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")

    if not os.path.exists(expert_file):
        print(f"Expert file not found for {task}, skipping.")
        continue
    if not os.path.exists(none_expert_file):
        print(f"None-expert file not found for {task}, skipping.")
        continue
    if not os.path.exists(json_filepath):
        print(f"JSON file not found for {task}, skipping.")
        continue

    # Load hidden states
    print(f"Loading hidden states for task: {task}")
    expert_data = np.load(expert_file).squeeze(axis=1)  # (num_expert_samples, num_layers, hidden_size)
    none_expert_data = np.load(none_expert_file).squeeze(axis=1)  # (num_none_expert_samples, num_layers, hidden_size)

    # Load JSON to identify inconsistent samples
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inconsistent_indices = []
    for idx, entry in enumerate(data.get("data", [])):
        ans_expert = entry.get(f"answer_{task}")
        ans_none = entry.get(f"answer_none_{task}")
        if ans_expert != ans_none:
            inconsistent_indices.append(idx)
    if not inconsistent_indices:
        print(f"No inconsistent samples for {task}, skipping.")
        continue

    # Check if the index is out of bounds
    if max(inconsistent_indices) >= expert_data.shape[0] or max(inconsistent_indices) >= none_expert_data.shape[0]:
        valid_indices = [i for i in inconsistent_indices if i < expert_data.shape[0] and i < none_expert_data.shape[0]]
        if not valid_indices:
            continue
        expert_data_diff = expert_data[valid_indices, ...]
        none_expert_data_diff = none_expert_data[valid_indices, ...]
    else:
        expert_data_diff = expert_data[inconsistent_indices, ...]
        none_expert_data_diff = none_expert_data[inconsistent_indices, ...]

    print(f"{task}: total samples = {expert_data.shape[0]}, inconsistent samples used = {expert_data_diff.shape[0]}")

    # Convert to float64 and clip
    expert_data_diff = expert_data_diff.astype(np.float64)
    none_expert_data_diff = none_expert_data_diff.astype(np.float64)
    expert_data_diff = np.clip(expert_data_diff, -1e6, 1e6)
    none_expert_data_diff = np.clip(none_expert_data_diff, -1e6, 1e6)

    all_expert_hidden_states.append(expert_data_diff)
    all_none_expert_hidden_states.append(none_expert_data_diff)

if not all_expert_hidden_states or not all_none_expert_hidden_states:
    raise ValueError("No inconsistent samples found across tasks.")

# Merge inconsistent samples
expert_hidden_states = np.concatenate(all_expert_hidden_states, axis=0)  # (N, num_layers, hidden_size)
none_expert_hidden_states = np.concatenate(all_none_expert_hidden_states, axis=0)  # 同样的形状

num_expert_samples, num_layers, hidden_size = expert_hidden_states.shape
num_none_expert_samples, _, _ = none_expert_hidden_states.shape
assert num_expert_samples == num_none_expert_samples, "Mismatch in sample counts!"

print(f"Total inconsistent samples: {num_expert_samples}")
print(f"Number of layers: {num_layers}, Hidden size per layer: {hidden_size}")

# ------------------ Construct PCA input samples ------------------
# For classification (expert vs. none-expert), we treat expert and none-expert as two categories,
# flatten them and merge them into X, with labels 1 and 0 respectively.
X_expert = expert_hidden_states.reshape(num_expert_samples, -1)
X_none_expert = none_expert_hidden_states.reshape(num_expert_samples, -1)

y_expert = np.ones(X_expert.shape[0], dtype=np.int32)
y_none_expert = np.zeros(X_none_expert.shape[0], dtype=np.int32)

# Merge data
X = np.vstack([X_expert, X_none_expert])  # 形状: (2*N, num_layers*hidden_size)
y = np.concatenate([y_expert, y_none_expert])

print(f"X shape: {X.shape}, y shape: {y.shape}")

# ------------------ PCA analysis ------------------
# Use 0.9 to automatically choose the number of principal components that explain 90% of variance
pca = PCA(n_components=0.9)
pca.fit(X)  # Use the entire dataset X for PCA

# Now, pca.n_components_ holds the actual number of components selected (an integer)
n_components_ = pca.n_components_
print(f"Selected number of components: {n_components_}")

# The shape of pca.components_ is (n_components_, num_layers*hidden_size)
# explained_variance_ratio_ represents the explained ratio of each component

# Calculate the importance of each feature (i.e., each neuron)
# Here, we use the weighted absolute loading coefficient:
# For each feature j, importance score = sum_{i=1}^{n_components_} (explained_variance_ratio_[i] * |loading[i, j]|)
importances = np.zeros(X.shape[1], dtype=np.float64)
for i in range(n_components_):
    importances += pca.explained_variance_ratio_[i] * np.abs(pca.components_[i, :])

# Reshape the importance vector to the original (num_layers, hidden_size) shape
importance_matrix = importances.reshape(num_layers, hidden_size)

# ------------------ Select Top 0.5% Neurons ------------------
total_neurons = num_layers * hidden_size
top_k = int(np.ceil((top_percentage / 100.0) * total_neurons))
top_k = max(top_k, 1)

# Range
sorted_indices = np.argsort(importance_matrix.flatten())[::-1]
top_indices = sorted_indices[:top_k]
top_neurons = [(int(idx // hidden_size), int(idx % hidden_size)) for idx in top_indices]

print(f"Top {top_percentage}% neurons based on PCA feature importance:")
for layer, neuron in top_neurons:
    imp = importance_matrix[layer, neuron]
    print(f"Layer {layer}, Neuron {neuron}, Importance: {imp:.6f}")

# ------------------ Save results ------------------
save_dir = os.path.join(current_path, "pca_results", model)
os.makedirs(save_dir, exist_ok=True)

# Save PCA matrix
importance_save_path = os.path.join(save_dir, f"pca_importance_{size}.npy")
np.save(importance_save_path, importance_matrix)
print(f"PCA importance matrix saved to {importance_save_path}")

# Save top neurons list
top_neurons_save_path = os.path.join(save_dir, f"top_{top_percentage}_percent_neurons_pca_{size}.json")
with open(top_neurons_save_path, 'w', encoding='utf-8') as f:
    json.dump(top_neurons, f, ensure_ascii=False, indent=4)
print(f"Top neurons saved to {top_neurons_save_path}")