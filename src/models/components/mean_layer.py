#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paveenhuang

Function Overview:
1) Compute the mean along the sample dimension for the original none-char hidden states and save it;
2) For each modified layer's hidden states file (named based on top and modified layer count), compute the mean and summarize it into a 4D array, then save it.
"""

import os
import numpy as np

# ------------------ Settings ------------------ #
TASKS = ["abstract_algebra", "anatomy", "econometrics", "global_facts", "jurisprudence"]

TOP_VALUES = [20, 640, 4096]

# Assume you want to iterate through layers [layer_start, layer_end)
layer_start = 0
layer_end = 31  # For example, 31 layers in total
model = "llama3"
size = "8B"

# Path settings
current_path = os.getcwd()
hidden_states_org_path = os.path.join(current_path, "hidden_states_v3", model)
hidden_states_mdf_path = os.path.join(current_path, "hidden_states_modified", model)
save_path = os.path.join(current_path, "hidden_states_mdf_mean", model)
os.makedirs(save_path, exist_ok=True)

print(f"Tasks: {TASKS}")
print(f"TOPs: {TOP_VALUES}")
print(f"Modify layers: {layer_start} ~ {layer_end} (total {layer_end - layer_start} layers)")
print(f"Model: {model}, size: {size}")

for task in TASKS:
    print(f"\n=== Processing Task: {task} ===")

    # ================ 1) Process original none‑char =================
    # Example of original filename: none_global_facts_global_facts_8B.npy
    data_none_char_org_filepath = os.path.join(hidden_states_org_path, f"none_{task}_{task}_{size}.npy")
    data_char_org_filepath = os.path.join(hidden_states_org_path, f"{task}_{task}_{size}.npy")
    if not os.path.exists(data_none_char_org_filepath):
        print(f"[Error] Original file not found: {data_none_char_org_filepath}")
        continue
    if not os.path.exists(data_char_org_filepath):
        print(f"[Error] Original file not found: {data_char_org_filepath}")
        continue

    data_none_char_org = np.load(data_none_char_org_filepath)  # (num_samples, len_tokens, model_layers, hidden_size)
    data_char_org = np.load(data_char_org_filepath)
    none_mean_org = np.mean(data_none_char_org, axis=0, keepdims=True)
    mean_org = np.mean(data_char_org, axis=0, keepdims=True)
    none_mean_org = np.squeeze(none_mean_org, axis=(0, 1)) 
    mean_org = np.squeeze(mean_org, axis=(0, 1))  # (model_layers, hidden_size)

    # Save
    orig_save_filename = f"{task}_{model}_{size}_org.npy"
    orig_save_filename_none = f"none_{task}_{model}_{size}_org.npy"
    out_org_path = os.path.join(save_path, orig_save_filename)
    out_org_path_none = os.path.join(save_path, orig_save_filename_none)
    np.save(out_org_path, mean_org)
    np.save(out_org_path_none, none_mean_org)
    print(f"[Saved original] {orig_save_filename} shape = {mean_org.shape}")
    print(f"[Saved original] {orig_save_filename_none} shape = {none_mean_org.shape}")

    # ================ 2) Process modified none‑char =================
    # Get related information from the shape of original mean_org
    model_layers, hidden_size = mean_org.shape  # mean_org (model_layers, hidden_size)
    mdf_layers = layer_end - layer_start

    # Allocate a 4D array to store the mean for all top+layer combinations
    modified_mean = np.zeros((len(TOP_VALUES), mdf_layers, model_layers, hidden_size), dtype=np.float32)

    # Double loop: Iterate through TOP and the "modified layers"
    for i_top, top in enumerate(TOP_VALUES):
        print(f"\n=== Processing Top: {top} ===")
        # For each layer in the range [layer_start, layer_end), load the corresponding file
        for layer_mod in range(layer_start, layer_end):
            i_layer = layer_mod - layer_start

            # Example modified filename: none_global_facts_global_facts_8B_640_29_30.npy
            # Represents the full hidden states after modifying layer 29
            filename = f"none_{task}_{task}_{size}_{top}_{layer_mod}_{layer_mod+1}.npy"
            data_none_char_mdf_filepath = os.path.join(hidden_states_mdf_path, filename)
            if not os.path.exists(data_none_char_mdf_filepath):
                print(f"[Warning] Modified file not found: {data_none_char_mdf_filepath}")
                continue
            data_none_char_mdf = np.load(data_none_char_mdf_filepath)
            mean_mdf = np.mean(data_none_char_mdf, axis=0)  #
            modified_mean[i_top, i_layer, :, :] = mean_mdf

            print(f"[Layer {layer_mod}] shape after mean & slice = {mean_mdf.shape}")

    # Save the final 4D array
    mod_save_filename = f"none_{task}_{model}_{size}_mdf.npy"
    mod_save_path = os.path.join(save_path, mod_save_filename)
    np.save(mod_save_path, modified_mean)

    print(f"[Saved modified] {mod_save_filename} with shape {modified_mean.shape}")
