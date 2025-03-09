#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 10:28:16 2025

This script loads replaced hidden states from a .npy file,
computes the mean across the sample dimension,
and saves the resulting tensor with shape (1,1,32,hidden_size).

@author: paveenhuang
"""

import os
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute mean hidden states over samples.")
    parser.add_argument("task", type=str, help="The task name (e.g. 'abstract_algebra').")
    parser.add_argument("size", type=str, help="Model size (e.g. '8B').")
    parser.add_argument("model", type=str, help="Model type (e.g. 'llama3').")
    parser.add_argument("start", type=int, default=0, help="Start layer index used in the replaced file.")
    parser.add_argument("end", type=int, default=1, help="End layer index used in the replaced file.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    task = args.task
    size = args.size
    model = args.model
    start = args.start
    end = args.end
    
    input_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_rpl/{model}"
    output_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_rpl_mean/{model}"
    input_file = os.path.join(input_dir, model, f"{task}_{size}_{start}_{end}.npy")
    
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load the replaced hidden states; expected shape: (num_samples, 1, num_layers, hidden_size)
    hs = np.load(input_file)
    print(f"Loaded replaced hidden states from {input_file} with shape {hs.shape}")

    # Compute the mean over the sample dimension (axis=0) and keep dimensions.
    hs_mean = np.mean(hs, axis=0, keepdims=True)
    print(f"Mean hidden states shape: {hs_mean.shape}")
    # Now hs_mean.shape should be (1, 1, num_layers, hidden_size), e.g. (1,1,32,4096)

    # Ensure output directory exists
    output_dir = os.path.join(output_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    # Save the mean hidden states with a new filename
    output_file = os.path.join(output_dir, f"mean_{task}_{size}_{start}_{end}.npy")
    np.save(output_file, hs_mean)
    print(f"Saved mean hidden states to: {output_file}")

if __name__ == "__main__":
    main()