#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:44:30 2025

@author: paveenhuang
"""

import os
import json
# import argparse
import numpy as np

# -------------------------------
# Parse command-line arguments
# -------------------------------
# parser = argparse.ArgumentParser(
#     description="Compute difference between char and none‑char mean hidden states (inconsistent samples only)"
# )
# parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
# parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
# args = parser.parse_args()

# model = args.model
# size = args.size

# Fixed parameters
model = "llama3"
size = "8B"

# -------------------------------
# Path definition
# -------------------------------
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
save_path = os.path.join(current_path, "hidden_states_v3_mean", model)
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

# -------------------------------
# Process each task to compute value_diff
# -------------------------------
for task in TASKS:
    print(f"Processing task: {task}")
    
    char_filepath = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
    none_char_filepath = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")
    
    char_data = np.load(char_filepath)
    none_char_data = np.load(none_char_filepath)
    if char_data.ndim == 4: # (samples,1,layers,hidden_size)
        char_data = np.squeeze(char_data, axis=1)
    if none_char_data.ndim == 4:
        none_char_data = np.squeeze(none_char_data, axis=1)
    
    # Calculate the mean of inconsistent samples for each task 
    # Resulting shape: (num_layers, hidden_size)
    char_mean = char_data.mean(axis=0)
    none_char_mean = none_char_data.mean(axis=0)
    
    ## Save
    save_file_char = os.path.join(save_path, f"{task}_{size}.npy")
    save_file_none = os.path.join(save_path, f"none_{task}_{size}.npy")
    np.save(save_file_char, char_mean)
    np.save(save_file_none, none_char_mean)
    
    print(f"Saved {save_file_char} and {save_file_none}")