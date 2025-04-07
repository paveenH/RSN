#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the mean hidden states for the 'no_role' condition
across all tasks and save a single mean .npy file.
"""

import os
import numpy as np
# import argparse

# Task list
TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_medicine",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# Parse command-line arguments
# parser = argparse.ArgumentParser(description="Compute no-role hidden state mean across tasks.")
# parser.add_argument("model", type=str, help="Model name (e.g., llama3)")
# parser.add_argument("size", type=str, help="Model size (e.g., 1B)")
# args = parser.parse_args()

model = "llama3_v3"
size = "8B"

# Directories
cwd = os.getcwd()
hidden_states_dir = os.path.join(cwd, "hidden_states_v3", model)
save_dir = os.path.join(cwd, "hidden_states_v3_mean", model)
os.makedirs(save_dir, exist_ok=True)

# Collect all no-role data
all_no_role = []

for task in TASKS:
    filepath = os.path.join(hidden_states_dir, f"no_role_{task}_{size}.npy")
    if not os.path.exists(filepath):
        print(f"[Warning] File not found, skipping: {filepath}")
        continue

    data = np.load(filepath)
    print(f"[Loaded] {task}: {data.shape[0]} samples")
    all_no_role.append(data)
print("data shape: data.shape")

# Compute and save mean
if all_no_role:
    combined = np.concatenate(all_no_role, axis=0)  # stack samples
    mean_hidden = combined.mean(axis=0, keepdims=True)  # mean over samples
    out_path = os.path.join(save_dir, f"no_role_all_mean_{size}.npy")
    np.save(out_path, mean_hidden)
    print(f"[Saved] No-role mean hidden states to: {out_path}")
else:
    print("No no-role data found across tasks; nothing to save.")