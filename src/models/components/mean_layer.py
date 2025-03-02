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
json_org_path = os.path.join(current_path, "answer", model)
json_mdf_path = os.path.join(current_path, "answer", f"{model}_layer")


def load_and_average(
        task: str, top: int, start: int, end: int
    ):
    # file name
    none_npy = os.path.join(hidden_states_path, f"none_{size}_{top}_{start}_{end}.npy")
    json_org_file = os.path.join(json_org_path, f"{task}_{size}_answers.json")
    json_mdf_file = os.path.join(json_mdf_path, f"{task}_{size}_answers_{top}_{start}_{end}.json")

    # check files
    if not os.path.exists(none_npy):
       print(f"[Warn] none file missing: {none_npy}")
       return None, None
    if not os.path.exists(json_org_file):
        print(f"[Warn] original json file missing: {json_org_file}")
        return None, None
    if not os.path.exists(json_mdf_file):
        print(f"[Warn] modified json file missing: {json_mdf_file}")
        return None, None

    # Load npy
    data_none = np.load(none_npy)  # shape: (num_samples, model_layers, hidden_size)

    # Load json
    with open(json_org_file, "r", encoding="utf-8") as jf:
        data_json_org = json.load(jf)
    with open(json_mdf_file, "r", encoding="utf-8") as jf:
        data_json_mdf = json.load(jf)

    # find inconsistent
    data_org = data_json_org.get("data", [])
    data_mdf = data_json_mdf.get("data", [])
    inconsistent_indices = []
    for idx, (entry_org, entry_mdf) in enumerate(zip(data_org, data_mdf)):
        ans_none_org = entry_org.get(f"answer_none_{task}")
        ans_none_mdf = entry_mdf.get(f"answer_none_{task}")
        if ans_none_org != ans_none_mdf:
            inconsistent_indices.append(idx)

    if not inconsistent_indices:
        print(f"[Info] No inconsistent sample for {task} (top={top}, layer={start}-{end})")
        return None, None

    data_none_inconsist = data_none[inconsistent_indices, ...]  # shape: (k, model_layers, hidden_size)
    mean_none = np.mean(data_none_inconsist, axis=0)

    return mean_none


def main():
    print(f"Tasks: {TASKS}")
    print(f"TOPs: {TOP_VALUES}")
    print(f"Modify layers: {layer_start}~{layer_end} (total {layer_end - layer_start} layers)")
    print(f"Model: {model}, size: {size}")

    for task in TASKS:
        print(f"\n=== Processing Task: {task} ===")
        guess_model_layers = None
        guess_hidden_size = None

        found_sample = False
        for top in TOP_VALUES:
            test_npy = os.path.join(hidden_states_path, f"none_{task}_{{task}}_{size}_{top}_{layer_start}_{layer_start+1}.npy")
            if os.path.exists(test_npy):
                sample_data = np.load(test_npy)
                # sample_data shape: (num_samples, model_layers, hidden_size)
                guess_model_layers = sample_data.shape[1]
                guess_hidden_size = sample_data.shape[2]
                found_sample = True
                break
        if not found_sample:
            print(f"[Error] Could not find any sample file for task={task} in first check. Skipping.")
            continue

        n_tops = len(TOP_VALUES)
        n_layers = layer_end - layer_start
        none_out = np.zeros((n_tops, n_layers, guess_model_layers, guess_hidden_size), dtype=np.float32)

        valid_mask = np.zeros((n_tops, n_layers), dtype=bool)

        for i_top, top in enumerate(TOP_VALUES):
            for layer_mod in range(layer_start, layer_end):
                i_layer = layer_mod - layer_start

                mean_none = load_and_average(task, top, layer_mod, layer_mod + 1)
                if mean_none is None:
                    print(f"[Warning] loss data: {task}-{top}-{layer_mod}")
                    continue
                none_out[i_top, i_layer, :, :] = mean_none
                valid_mask[i_top, i_layer] = True

        none_filename = f"none_{task}_{model}_{size}.npy"
        np.save(os.path.join(save_path, none_filename), none_out)

        print(f"[Saved] {none_filename}")
        print(f"  final shape => none_out: {none_out.shape}")
        print(f"  valid ratio => {np.sum(valid_mask)}/{valid_mask.size}")

if __name__ == "__main__":
    main()