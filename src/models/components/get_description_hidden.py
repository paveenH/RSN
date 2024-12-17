#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:18:21 2024

@author: paveenhuang
"""

import json
import argparse
import os
import numpy as np
from tqdm import tqdm

from vicuna import VicundaModel

# Define the path to the JSON files
PATH = "/data2/paveen/RolePlaying/src/models/components/"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
parser.add_argument("size", type=str, help="The size of the model (e.g., 13B).")
args = parser.parse_args()

size = args.size

# Define model path
model_path = f"/data2/paveen/RolePlaying/shared/llama3/{size}"
json_path = os.path.join(PATH, "description.json")

# Initialize VicundaModel
vc = VicundaModel(model_path=model_path)
template = "Here is a description of an expert: {context}.\nQuestion: Based on the description, please choose the identity of this character from the following options:\nA) management expert\nB) medical genetics expert\nAnswer:"

# Load JSON data
print(f"Loading JSON data from {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)


hidden_states_storage = []
print("Starting hidden state extraction...")
for idx, sample in enumerate(tqdm(data, desc="Extracting Hidden States")):
    context = sample.get("text", "")
    if not context:
        print(f"Sample {idx} is missing 'text' field. Skipping.")
        continue
    prompt = template.format(context=context)
    results = vc.get_hidden_states(prompt=prompt, temptype="description")  

    if any(r is None for r in results):
        print(f"Sample {idx} has missing hidden states. Skipping.")
        continue

    hidden_states_list = []
    for pos_layers in results:  # pos_layers is a list of length num_layers, each element is a (hidden_size,) vector
        pos_array = np.stack(pos_layers, axis=0)  # (num_layers, hidden_size)
        hidden_states_list.append(pos_array)

    hidden_states_array = np.stack(hidden_states_list, axis=0)
    hidden_states_storage.append(hidden_states_array)

print("Finished extracting hidden states.")

# Save hidden states
save_dir = os.path.join("/data2/paveen/RolePlaying/src/models/components/hidden_states", f"hidden_states_{size}_description")
os.makedirs(save_dir, exist_ok=True)

print("Saving hidden states to disk...")
if not hidden_states_storage:
    print("No hidden states collected. Skipping save.")
else:
    hs_array = np.stack(hidden_states_storage, axis=0)
    # Define the save path
    save_path = os.path.join(save_dir, "description_hidden_states.npy")
    # Save as .npy file
    np.save(save_path, hs_array)
    print(f"Saved hidden states for description to {save_path}")

print("All hidden states have been saved successfully.")