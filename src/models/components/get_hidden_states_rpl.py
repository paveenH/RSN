#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:05:00 2025

@author: paveenhuang

Script to:
1) Load "None" and "Expert" hidden states from .npy files
2) For each sample, build a "None" prompt
3) Replace the [start:end) layers' hidden states with "Expert"
4) Extract the final hidden states (pos1 = last token) after replacement
5) Save the replaced hidden states to disk
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from vicuna import VicundaModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Get replaced hidden states for each sample.")
    parser.add_argument("task", type=str, help="The name of the task to process.")
    parser.add_argument("size", type=str, help="Model size (e.g. '13B').")
    parser.add_argument("model", type=str, help="Model type (e.g. 'llama3').")
    parser.add_argument("--start", type=int, default=0, help="Start layer index for replacement (inclusive).")
    parser.add_argument("--end", type=int, default=1, help="End layer index for replacement (exclusive).")
    return parser.parse_args()

def main():
    args = parse_arguments()
    task = args.task
    size = args.size
    model_name = args.model

    start_layer = args.start
    end_layer = args.end

    # 1) Load model
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    vc = VicundaModel(model_path=model_path)
    template = vc.template
    print(f"Loaded model: {model_name}, size={size}")
    print(f"Template:\n{template}")

    # 2) Prepare data
    mmlu_path = "/data2/paveen/RolePlaying/src/models/components/mmlu"
    json_path = os.path.join(mmlu_path, f"{task}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file '{json_path}' not found.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")

    # 3) Load original none & expert hidden states
    hs_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v5/{model_name}"
    none_hs_path = os.path.join(hs_dir, f"none_{task}_{size}.npy")
    expert_hs_path = os.path.join(hs_dir, f"{task}_{size}.npy")

    if not (os.path.isfile(none_hs_path) and os.path.isfile(expert_hs_path)):
        raise FileNotFoundError(f"Cannot find HS npy files: {none_hs_path} or {expert_hs_path}")

    none_array = np.load(none_hs_path)   # (num_samples, 1, 33, hidden_size)
    expert_array = np.load(expert_hs_path)
    if none_array.shape != expert_array.shape:
        raise ValueError("None & Expert hidden states shape mismatch.")

    # Remove embedding layer (index=0)
    none_array = none_array[:, :, 1:, :]   # (num_samples, 1, 32, hidden_size)
    expert_array = expert_array[:, :, 1:, :]

    num_samples, _, num_layers, hidden_size = none_array.shape
    print(f"After removing embedding layer => shape: {none_array.shape}")
    print(f"  #samples={num_samples}, #layers={num_layers}, hidden={hidden_size}")

    # Check layer range
    if not (0 <= start_layer < end_layer <= num_layers):
        raise ValueError(f"Invalid layer range: [start={start_layer}, end={end_layer}), must be in [0, {num_layers})")

    # 4) Create a storage for replaced hidden states
    #    We'll store final shape => (num_samples, num_layers, hidden_size)
    #    or possibly (num_samples, 1, num_layers, hidden_size) if you want same shape as code1
    replaced_hs_list = []

    # 5) Iterate each sample
    print(f"Replacing None->Expert in layers [{start_layer}:{end_layer}), extracting final HS...")
    for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
        context = sample.get("text", "")
        if not context:
            continue

        if idx >= num_samples:
            print(f"Sample idx={idx} out of range for HS array (size={num_samples}), break.")
            break

        # Build None prompt
        none_character = f"none {task.replace('_',' ')}"
        prompt = template.format(character=none_character, context=context)

        # (num_layers, hidden_size)
        none_hs = none_array[idx, 0]
        expert_hs = expert_array[idx, 0]

        # replace_matrices (num_layers, hidden_size)
        replace_matrices = none_hs.copy()
        replace_matrices[start_layer:end_layer] = expert_hs[start_layer:end_layer]

        # 6) get replaced hidden states => returns a list (pos1, pos2, ...)
        #    but if temptype='abcde', typically just [pos1], shape = [num_layers, hidden_size]
        replaced_positions = vc.get_hidden_states_rpl(
            prompt=prompt,
            replace_matrices=replace_matrices,
            start=start_layer,
            end=end_layer,
        )
        # Usually replaced_positions[0] is [num_layers, hidden_size]
        if not replaced_positions or replaced_positions[0] is None:
            # something went wrong
            continue

        final_hs = replaced_positions[0]  # shape (num_layers, hidden_size)
        # If you want (1, num_layers, hidden_size), can reshape:
        final_hs = np.expand_dims(final_hs, axis=0)
        # now shape is (1, num_layers, hidden_size)

        replaced_hs_list.append(final_hs)  # we'll stack them later

    # 7) Save replaced hidden states
    if not replaced_hs_list:
        print("No replaced hidden states were collected.")
        return

    replaced_arr = np.stack(replaced_hs_list, axis=0)  
    # shape => (num_samples, 1, num_layers, hidden_size)

    # Save dir
    save_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_replaced/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    out_file = os.path.join(save_dir, f"replaced_{task}_{size}_{start_layer}_{end_layer}.npy")
    np.save(out_file, replaced_arr)
    print(f"Saved replaced hidden states to: {out_file}")
    print("All done!")


if __name__ == "__main__":
    main()