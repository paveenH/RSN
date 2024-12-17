#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:18:21 2024

@author: paveenhuang
"""

import json
import argparse
import os
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

# Define JSON file path
json_path = os.path.join(PATH, "description.json")


# Initialize VicundaModel
vc = VicundaModel(model_path=model_path)
template = "Here is a description of an expert: {context}.\nQuestion: Based on the description, please choose the identity of this character from the following options:\nA) management expert\nB) medical genetics expert\nAnswer:"

# Load JSON data
print(f"Loading JSON data from {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples loaded: {len(data)}")

# Define characters and labels
characters = ["management expert", "medical genetics expert"]
label_mapping = {"A": 0, "B": 1}

# Initialize storage for generated outputs
generated_outputs_storage = []

# Counters for accuracy
correct_predictions = 0
total_predictions = 0

# Iterate through each sample
print("Starting generation of outputs and evaluating accuracy...")
for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
    context = sample.get("text", "")
    true_label = sample.get("label", None)
    if not context or true_label is None:
        print(f"Sample {idx} is missing 'text' or 'label' field. Skipping.")
        continue

    # Generate prompt
    prompt = template.format(context=context)

    # Generate output
    generated_output = vc.generate([prompt])[0]  # generate expects a list and returns a list

    # Parse the generated output to extract the answer (A or B)
    # Assuming the model outputs something like "A", "A)", "B", "B)", or starts with A or B
    # We'll extract the first character that is A or B
    parsed_answer = None
    for char in generated_output.strip():
        if char.upper() in label_mapping:
            parsed_answer = char.upper()
            break

    if parsed_answer is not None:
        predicted_label = label_mapping[parsed_answer]
        is_correct = predicted_label == true_label
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
    else:
        predicted_label = None
        is_correct = False
        print(f"Sample {idx}: Unable to parse answer from generated output: '{generated_output}'")

    # Append to storage
    generated_outputs_storage.append({
        "text": context,
        "label": true_label,
        "generated_output": generated_output,
        "predicted_label": predicted_label,
        "is_correct": is_correct
    })

print("Finished generating outputs and evaluating accuracy.")

# Calculate accuracy
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions} correct)")
else:
    print("No valid predictions were made.")

# Save the generated outputs to a JSON file
generated_outputs_path = os.path.join(PATH, "description_outputs.json")
with open(generated_outputs_path, "w", encoding="utf-8") as f:
    json.dump(generated_outputs_storage, f, ensure_ascii=False, indent=4)
print(f"Generated outputs have been saved to {generated_outputs_path}")    

    

        
        
        