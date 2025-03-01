#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:27:16 2025

Reading difference matrices, modifying certain layers, 
then obtaining the hidden states from a Vicuna-like model.

@author: paveenhuang
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from vicuna import VicundaModel


def parse_arguments():
    """
    Parse command-line arguments for:
      - task
      - model
      - size
      - top (top-K dimensions per layer to keep)
      - alpha (scaling factor)
      - start (start layer index)
      - end (end layer index, exclusive)
    """
    parser = argparse.ArgumentParser(description="Modify certain layers' hidden states and extract them.")
    parser.add_argument(
        "task_size", type=str, help="Combined argument with task, model, size, top, alpha, start, end separated by spaces."
    )
    args = parser.parse_args()

    try:
        task, model_name, size, top, alpha, start, end = args.task_size.split()
    except ValueError:
        raise ValueError(
            "The task_size parameter should contain seven parts: "
            "task, model, size, top, alpha, start, end (separated by spaces)."
        )

    return task, model_name, size, int(top), float(alpha), int(start), int(end)


def main():
    task, model_name, size, top, alpha, start, end = parse_arguments()

    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    json_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    matrix_path = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_mean/{model_name}"

    save_dir = os.path.join(f"/data2/paveen/RolePlaying/src/models/components/hidden_states_modified/{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        data_char_diff = np.load(os.path.join(matrix_path, f"all_mean_{size}.npy"))  # (1,1,layers,hidden_size)
        data_none_char_diff = np.load(os.path.join(matrix_path, f"none_all_mean_{size}.npy"))  # (1,1,layers,hidden_size)
    except FileNotFoundError as e:
        print(f"Error loading difference matrices: {e}")
        return

    # Compute the difference matrix and remove the batch dimension; exclude the embedding layer (index=0)
    char_differences = (data_char_diff - data_none_char_diff).squeeze(0).squeeze(0)  # (layers, hidden_size)
    num_layers = char_differences.shape[0]
    char_differences = char_differences[1:] * alpha  # => (layers-1, hidden_size)

    num_layers = char_differences.shape[0]

    start = max(0, min(start, num_layers - 1))
    end = max(start + 1, min(end, num_layers))

    print(f"Total transformer layers (excluding embedding): {num_layers}")
    print(f"Will apply modification on layer indices [{start}, {end}).")
    print(f"Top-K = {top}, alpha = {alpha}")

    # Perform Top-K filtering on the specified layer range, and set other layers to 0
    if top >= 0:
        for layer_idx in range(num_layers):
            if start <= layer_idx < end:
                layer_diff = char_differences[layer_idx]
                top_indices = np.argsort(np.abs(layer_diff))[-top:]
                mask = np.zeros_like(layer_diff, dtype=bool)
                mask[top_indices] = True
                char_differences[layer_idx] = np.where(mask, layer_diff, 0)
            else:
                char_differences[layer_idx] = 0

    print(f"char_differences shape after top-{top} masking: {char_differences.shape}")

    vc = VicundaModel(model_path=model_path)
    template = vc.template

    # Load json
    if not os.path.isfile(json_path):
        print(f"JSON file {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {json_path}.")

    # TODO: Need to loop in characters.
    task_name = task.replace("_", " ")
    # character = f"none {task_name}"
    characters = [f"none {task_name}", task_name]
    # Generate prompt for each sample and get the modified hidden state
    hidden_states_storage = {character: [] for character in characters}

    for idx, sample in enumerate(tqdm(data, desc="Extracting Hidden States")):
        context = sample.get("text", "")
        if not context:
            continue

        for character in characters:
            prompt = template.format(character=character, context=context)

            # modified hidden states (last token)
            hs_mdf = vc.get_hidden_states_mdf(
                prompt=prompt, diff_matrices=char_differences
            )  # return [token_hs], pos1 = last_token

            # hs_mdf is a list with only one element (because of pos1), which stores the hidden states of each layer.
            # shape [num_layers, hidden_size]
            if not hs_mdf or hs_mdf[0] is None:
                continue
            layer_hidden = hs_mdf[0]  # shape (num_layers, hidden_size)
            hidden_states_storage[character].append(layer_hidden)

    # save hidden states
    for character, hs_list in hidden_states_storage.items():
        if hs_list:
            all_hidden = np.stack(hs_list, axis=0)
            character_safe = character.replace(" ", "_")
            out_name = f"{character_safe}_{task}_{size}_{top}_{start}_{end}.npy"
            save_path = os.path.join(save_dir, out_name)
            np.save(save_path, all_hidden)
            print(f"Saved modified hidden states for '{character}' to {save_path}")
        else:
            print("No hidden states were collected. Nothing to save.")


if __name__ == "__main__":
    main()
