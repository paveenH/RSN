#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:12:52 2025

@author: paveenhuang

Objective: Use XGBoost to classify expert vs. none-expert based on hidden states
and extract the top 0.5% most important neurons.
"""

import os
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ------------------ Parameter parsing ------------------
parser = argparse.ArgumentParser(description="Train XGBoost classifier on inconsistent samples to extract top neurons")
parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
parser.add_argument("size", type=str, help="Size of the model (e.g., 1B or 3B)")
parser.add_argument("top_percentage", type=float, help="Top percentage of neurons to select based on feature importance (e.g., 0.5 for top 0.5%)")
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
    expert_data = np.load(expert_file)  # (num_expert_samples, 1, num_layers, hidden_size)
    none_expert_data = np.load(none_expert_file)  # (num_none_expert_samples, 1, num_layers, hidden_size)

    expert_data = expert_data.squeeze(axis=1)  # (num_expert_samples, num_layers, hidden_size)
    none_expert_data = none_expert_data.squeeze(axis=1)  # (num_none_expert_samples, num_layers, hidden_size)

    # Load JSON answer, screening for inconsistent samples
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

    print(f"{task}: total samples={expert_data.shape[0]}, inconsistent samples used={expert_data_diff.shape[0]}")

    # Convert to float64 and clip to prevent numerical anomalies
    expert_data_diff = expert_data_diff.astype(np.float64)
    none_expert_data_diff = none_expert_data_diff.astype(np.float64)
    expert_data_diff = np.clip(expert_data_diff, -1e6, 1e6)
    none_expert_data_diff = np.clip(none_expert_data_diff, -1e6, 1e6)

    all_expert_hidden_states.append(expert_data_diff)
    all_none_expert_hidden_states.append(none_expert_data_diff)

if not all_expert_hidden_states or not all_none_expert_hidden_states:
    raise ValueError("No inconsistent samples found across tasks.")

# Merge inconsistent samples from all tasks
expert_hidden_states = np.concatenate(all_expert_hidden_states, axis=0)  # (N, num_layers, hidden_size)
none_expert_hidden_states = np.concatenate(all_none_expert_hidden_states, axis=0)  # Same shape

# Make sure the number of samples is consistent
num_expert_samples, num_layers, hidden_size = expert_hidden_states.shape
num_none_expert_samples, _, _ = none_expert_hidden_states.shape
assert num_expert_samples == num_none_expert_samples, "Mismatch in sample counts!"

print(f"Total inconsistent samples: {num_expert_samples}")
print(f"Number of layers: {num_layers}, Hidden size per layer: {hidden_size}")

# ------------------ Constructing training data ------------------
# Construct labels for classification: expert -> 1, none-expert -> 0
# Flatten the hidden states of each sample into a vector: shape (N, num_layers * hidden_size)
X_expert = expert_hidden_states.reshape(num_expert_samples, -1)
X_none_expert = none_expert_hidden_states.reshape(num_expert_samples, -1)

y_expert = np.ones(X_expert.shape[0], dtype=np.int32)
y_none_expert = np.zeros(X_none_expert.shape[0], dtype=np.int32)

# Merge data
X = np.vstack([X_expert, X_none_expert])
y = np.concatenate([y_expert, y_none_expert])

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------ Train XGBoost model ------------------

# xgb_model = xgb.XGBClassifier(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     use_label_encoder=False,
#     eval_metric="logloss"
# )

xgb_model = xgb.XGBClassifier(
    tree_method="hist",
    max_bin=256,
    max_depth=6,
    n_jobs=2, 
    subsample=0.5,
    colsample_bytree=0.5,
)

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# Prediction and acc
y_pred = xgb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"XGBoost classification accuracy: {acc:.4f}")

# ------------------ Extract Feature Importance ------------------
importance = xgb_model.feature_importances_  # shape: (num_layers * hidden_size,)

# Reshape to  (num_layers, hidden_size)
importance_matrix = importance.reshape(num_layers, hidden_size)

# ------------------ Choose Top 0.5%  Neurons ------------------
total_neurons = num_layers * hidden_size
top_k = int(np.ceil((top_percentage / 100.0) * total_neurons))
top_k = max(top_k, 1)

# Sort by importance (most important to least important)
sorted_indices = np.argsort(importance.flatten())[::-1]
top_indices = sorted_indices[:top_k]
top_neurons = [(int(idx // hidden_size), int(idx % hidden_size)) for idx in top_indices]

print(f"Top {top_percentage}% neurons based on XGBoost feature importance:")
for layer, neuron in top_neurons:
    imp = importance_matrix[layer, neuron]
    print(f"Layer {layer}, Neuron {neuron}, Importance: {imp:.6f}")

# ------------------ Save results ------------------
save_dir = os.path.join(current_path, "xgboost_results", model)
os.makedirs(save_dir, exist_ok=True)

# Save feature importance matrix
importance_save_path = os.path.join(save_dir, f"feature_importance_{size}.npy")
np.save(importance_save_path, importance_matrix)
print(f"Feature importance matrix saved to {importance_save_path}")

# Save top neurons list
top_neurons_save_path = os.path.join(save_dir, f"top_{top_percentage}_percent_neurons_xgb_{size}.json")
with open(top_neurons_save_path, 'w', encoding='utf-8') as f:
    json.dump(top_neurons, f, ensure_ascii=False, indent=4)
print(f"Top neurons saved to {top_neurons_save_path}")