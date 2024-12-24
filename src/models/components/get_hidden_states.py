#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:38:53 2024

@author: paveenhuang
"""

import json
import argparse
import os
import numpy as np
from tqdm import tqdm

from vicuna import VicundaModel


# Define the path to the JSON files
PATH = "/data2/paveen/RolePlaying/src/models/components/mmlu"

# Parse combined arguments
parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
parser.add_argument("task_size", type=str, help="The task and size as a combined argument.")
args = parser.parse_args()

# Split task and size
task, size = args.task_size.split()


# Define model path
model_path = f"/data2/paveen/RolePlaying/shared/llama3/{size}"   

# Define JSON file path
json_path = os.path.join(PATH, f"{task}.json")

# Define save directory for hidden states
save_dir = os.path.join("/data2/paveen/RolePlaying/src/models/components/hidden_states_abcde")
os.makedirs(save_dir, exist_ok=True)

# Initialize VicundaModel
vc = VicundaModel(model_path=model_path)
# template = "You are a {character}, You are a {character}, You are a {character}, would you answer the following question with A, B, C or D? \n Question: {context}\n Answer: "
template = "Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you are a honest {character} expert, your answer among{{A, B, C, D, E}} is: "

# Define the list of characters
task_name = task.replace('_', ' ')
characters = [f"none {task_name}", task_name]

# Load JSON data
print(f"Loading JSON data from {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples loaded: {len(data)}")

# Initialize storage for hidden states
hidden_states_storage = {character: [] for character in characters}

# Iterate through each sample
print("Starting hidden state extraction...")
for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
    context = sample.get("text", "")
    if not context:
        print(f"Sample {idx} is missing 'text' field. Skipping.")
        continue

    for character in characters:
        # Generate prompt
        prompt = template.format(character=character, context=context)

        # Extract hidden states
        hidden_states = vc.get_hidden_states(prompt=prompt, 
                                             character=character,
                                             temptype = "abcde")

        # Check if all positions were found
        if any(pos is None for pos in hidden_states):
            print(f"Sample {idx} for character '{character}' has missing hidden states. Skipping.")
            continue

        hidden_states_array = np.stack([np.stack(pos, axis=0) for pos in hidden_states], axis=0)

        # Append to storage
        hidden_states_storage[character].append(hidden_states_array)

print("Finished extracting hidden states.")

# Save hidden states
print("Saving hidden states to disk...")
for character, hs_list in hidden_states_storage.items():
    if not hs_list:
        print(f"No hidden states collected for character '{character}'. Skipping save.")
        continue
    # Convert list to numpy array: shape (num_samples, 6, num_layers, hidden_size)
    hs_array = np.stack(hs_list, axis=0)
    # Define save path
    character_safe = character.replace(' ', '_')
    save_path = os.path.join(save_dir, f"{character_safe}_{task}_{size}.npy")
    # Save as .npy file
    np.save(save_path, hs_array)
    print(f"Saved hidden states for '{character}' to {save_path}")

print("All hidden states have been saved successfully.")