#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 10:28:16 2025

This script loads replaced hidden states from multiple .npy files,
computes the mean across the sample dimension,
and saves the resulting tensor with shape (1,1,32,hidden_size).

@author: paveenhuang
"""

import os
import numpy as np

# Define task list, model, size
TASKS = [
    "abstract_algebra",
    "anatomy",
    "global_facts",
    "econometrics",
    "jurisprudence"
]
MODEL = "llama3"
SIZE = "8B"

# Directory settings
INPUT_DIR = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_rpl/{MODEL}"
OUTPUT_DIR = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_rpl_mean/{MODEL}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all start-end combinations
START = 0
END = 31 

pairs = [(i, i + 1) for i in range(START, END)]

for task in TASKS:
    aggregated_list = []
    print(f"Processing task: {task}")
    for (start_layer, end_layer) in pairs:
        file_name = f"{task}_{SIZE}_{start_layer}_{end_layer}.npy"
        file_path = os.path.join(INPUT_DIR, file_name)
        if not os.path.isfile(file_path):
            print(f"Skipping {file_path} (file not found)")
            continue
        # Load the mean file with the expected shape (1,1,32,hidden_size)
        hs_mean = np.load(file_path)
        print(f"Loaded {file_path} with shape {hs_mean.shape}")
        # Remove the first two batch dimensions to make its shape become (32, hidden_size).
        hs_mean = np.squeeze(hs_mean, axis=0)  # (1,32,hidden_size)
        hs_mean = np.squeeze(hs_mean, axis=0)  # (32, hidden_size)
        aggregated_list.append(hs_mean)
    if not aggregated_list:
        print(f"No files aggregated for task {task}.")
        continue
    # Stack all files into a matrix with the shape (num_pairs, num_layers, hidden_size)
    aggregated_matrix = np.stack(aggregated_list, axis=0)
    print(f"Aggregated matrix shape for task {task}: {aggregated_matrix.shape}")
    # Save the aggregated matrix
    output_file = os.path.join(OUTPUT_DIR, f"combined_{task}_{SIZE}.npy")
    np.save(output_file, aggregated_matrix)
    print(f"Saved aggregated matrix for task {task} to: {output_file}")

print("All tasks completed!")