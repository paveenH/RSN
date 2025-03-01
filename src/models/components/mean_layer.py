#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paveenhuang
"""

import os
import numpy as np
import json

# ------------------ Settings ------------------ #
TASKS = [
    "abstract_algebra",
    "anatomy",
    "econometrics",
    "global_facts",
    "jurisprudence"
]

TOP_VALUES = [20, 640, 4096]

layer_start = 0
layer_end = 31  
model = "llama3"
size = "8B"

# Path
current_path = os.getcwd()
hidden_states_path = os.path.join(current_path, "hidden_states_modified", model)
# output path
save_path = os.path.join(current_path, "hidden_states_layer_mean", model)
os.makedirs(save_path, exist_ok=True)
# JSON path 
json_path = os.path.join(current_path, "answer", f"{model}_layer")


def load_and_average(
    task: str, top: int, start: int, end: int
):
    # file name
    char_npy = os.path.join(hidden_states_path, f"{task}_{size}_{top}_{start}_{end}.npy")
    none_npy = os.path.join(hidden_states_path, f"none_{size}_{top}_{start}_{end}.npy")
    json_file = os.path.join(json_path, f"{task}_{size}_answers_{top}_{start}_{end}.json")

    # check files
    if not os.path.exists(char_npy):
        print(f"[Warn] char file missing: {char_npy}")
        return None, None
    if not os.path.exists(none_npy):
        print(f"[Warn] none file missing: {none_npy}")
        return None, None
    if not os.path.exists(json_file):
        print(f"[Warn] json file missing: {json_file}")
        return None, None

    # Load npy
    data_char = np.load(char_npy)  # shape: (num_samples, model_layers, hidden_size)
    data_none = np.load(none_npy)  # shape: (num_samples, model_layers, hidden_size)

    # Load json
    with open(json_file, "r", encoding="utf-8") as jf:
        data_json = json.load(jf)

    # find inconsistent
    inconsistent_indices = []
    for idx, entry in enumerate(data_json.get("data", [])):
        ans_none = entry.get(f"answer_none_{task}")
        ans_char = entry.get(f"answer_{task}")
        if ans_none != ans_char:
            inconsistent_indices.append(idx)
    if not inconsistent_indices:
        print(f"[Info] No inconsistent sample for {task} (top={top}, layer={start}-{end})")
        return None, None

    data_char_inconsist = data_char[inconsistent_indices, ...]  # shape: (k, model_layers, hidden_size)
    data_none_inconsist = data_none[inconsistent_indices, ...]  # shape: (k, model_layers, hidden_size)

    mean_char = np.mean(data_char_inconsist, axis=0)
    mean_none = np.mean(data_none_inconsist, axis=0)

    return mean_char, mean_none


def main():
    print(f"Tasks: {TASKS}")
    print(f"TOPs: {TOP_VALUES}")
    print(f"Modify layers: {layer_start}~{layer_end} (total {layer_end-layer_start} layers)")
    print(f"Model: {model}, size: {size}")

    for task in TASKS:
        print(f"\n=== Processing Task: {task} ===")
        guess_model_layers = None
        guess_hidden_size = None

        found_sample = False
        for top in TOP_VALUES:
            test_npy = os.path.join(hidden_states_path, f"{task}_{size}_{top}_{layer_start}_{layer_start+1}.npy")
            if os.path.exists(test_npy):
                sample_data = np.load(test_npy)
                # shape: (num_samples, model_layers, hidden_size)
                guess_model_layers = sample_data.shape[1]
                guess_hidden_size = sample_data.shape[2]
                found_sample = True
                break
        if not found_sample:
            print(f"[Error] Could not find any sample file for task={task} in first check. Skipping.")
            continue

        # Prepare to allocate a large array => (3, 31, model_layers, hidden_size)
        n_tops = len(TOP_VALUES)
        n_layers = layer_end - layer_start
        char_out = np.zeros((n_tops, n_layers, guess_model_layers, guess_hidden_size), dtype=np.float32)
        none_out = np.zeros((n_tops, n_layers, guess_model_layers, guess_hidden_size), dtype=np.float32)

        # Also record which locations have no data => to distinguish
        valid_mask = np.zeros((n_tops, n_layers), dtype=bool)

        for i_top, top in enumerate(TOP_VALUES):
            for layer_mod in range(layer_start, layer_end):
                i_layer = layer_mod - layer_start

                mean_char, mean_none = load_and_average(task, top, layer_mod, layer_mod+1)
                if mean_char is None or mean_none is None:
                    print(f"[Waring] loss data: {task}-{top}-{layer_mod}")
                    continue
                else:
                    # shape = (model_layers, hidden_size)
                    char_out[i_top, i_layer, :, :] = mean_char
                    none_out[i_top, i_layer, :, :] = mean_none
                    valid_mask[i_top, i_layer] = True

        # Generate file name
        #   {task}_{model}_{size}_layerMean_char.npy
        #   {task}_{model}_{size}_layerMean_none.npy
        char_filename = f"{task}_{model}_{size}_char_out.npy"
        none_filename = f"{task}_{model}_{size}_none_out.npy"

        np.save(os.path.join(save_path, char_filename), char_out)
        np.save(os.path.join(save_path, none_filename), none_out)


        print(f"[Saved] {char_filename}, {none_filename}")
        print(f"  final shape => char_out: {char_out.shape}, none_out: {none_out.shape}")
        print(f"  valid ratio => {np.sum(valid_mask)}/{valid_mask.size}")

if __name__ == "__main__":
    main()