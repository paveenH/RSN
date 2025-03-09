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
END = 32  
START_END_PAIRS = [(i, i + 1) for i in range(START, END)]

for task in TASKS:
    for start, end in START_END_PAIRS:
        input_file = os.path.join(INPUT_DIR, f"{task}_{SIZE}_{start}_{end}.npy")
        output_file = os.path.join(OUTPUT_DIR, f"mean_{task}_{SIZE}_{start}_{end}.npy")

        if not os.path.isfile(input_file): 
            print(f"Skipping: File not found {input_file}")
            continue

        # Read hidden state
        hs = np.load(input_file)
        print(f"Loaded {input_file} with shape {hs.shape}")

        # Calculate the mean over the sample dimension
        hs_mean = np.mean(hs, axis=0, keepdims=True)
        print(f"Mean hidden states shape: {hs_mean.shape}")  # 预期形状: (1,1,32,hidden_size)

        # Save mean file
        np.save(output_file, hs_mean)
        print(f"Saved mean hidden states to: {output_file}")

print("All tasks completed!")