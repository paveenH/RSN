#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:15:20 2024

@author: paveenhuang
"""

import json
import argparse
import os
from tqdm import tqdm
import numpy as np
from vicuna import VicundaModel
from get_answer import cleaning, extract_full_correct_text, handle_invalid_answer


# Get parser parameters
parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
parser.add_argument("task_size_model", type=str, help="The task and size as a combined argument.")
args = parser.parse_args()

task, size, model_name = args.task_size_model.split()
print(f"Task: {task}")
print(f"Size: {size}")
print(f"Model Name: {model_name}")

# Path definition
model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
json_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
matrix_path = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3/{model_name}"

# Load data
data_char_diff = np.load(f'{matrix_path}/all_mean_{size}.npy')       # (1,1,layers,hidden_size)
data_none_char_diff =  np.load(f'{matrix_path}/none_all_mean_{size}.npy') # (1,1,layers,hidden_size)
char_differences = data_char_diff - data_none_char_diff               # (1,1,layers,hidden_size)
char_differences = char_differences.squeeze(0).squeeze(0)             # (layers, hidden_size)
char_differences = char_differences[1:]                               # exclude embedding layer

# Calculate hidden_size and top
hidden_size = char_differences.shape[1]  # Determine hidden_size dynamically
top = hidden_size // 200                 # Retain top neurons per layer
save_dir = os.path.join(f"/data2/paveen/RolePlaying/src/models/components/answer_modified_{top}")
os.makedirs(save_dir, exist_ok=True)

# Debugging: print calculated values
print(f"Hidden size: {hidden_size}, Top neurons to retain per layer: {top}")

if top > 0:
    print(f"Top {top} calculation begin.")
    for layer_idx in range(char_differences.shape[0]): 
        layer_diff = char_differences[layer_idx]  # (hidden_size,)
        top_indices = np.argsort(np.abs(layer_diff))[-top:]   # Top N
        mask = np.zeros_like(layer_diff, dtype=bool)
        mask[top_indices] = True
        char_differences[layer_idx] = np.where(mask, layer_diff, 0)

# Debug
print(f"data_char_diff shape: {data_char_diff.shape}")
print(f"data_none_char_diff shape: {data_none_char_diff.shape}")
print(f"char_differences shape: {char_differences.shape}")

# Initialize model
vc = VicundaModel(model_path=model_path)
template = vc.template

# characters
task_name = task.replace('_', ' ')
characters = [f"none {task_name}", f"{task_name}"]

# Load json
print(f"Loading JSON data from {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Total samples loaded: {len(data)}")

# acc
accuracy_counts = {
    character: {
        "correct": 0, 
        "total": 0, 
        "E_count": 0,
        "invalid": 0
    } 
    for character in characters
}

label_mapping = ["A", "B", "C", "D"]

print("Starting answer generation and accuracy calculation...")

for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
    context = sample.get("text", "")
    true_label_int = sample.get("label", -1)
    if not (0 <= true_label_int < len(label_mapping)):
        continue
    true_label = label_mapping[true_label_int]

    # Used for text matching in subsequent invalid fallback
    true_label_text = extract_full_correct_text(context, true_label_int)

    for character in characters:
        # Construct prompt
        prompt = template.format(character=character, context=context)

        # Generate answer: call regenerate and append the difference matrix
        generated_output = vc.regenerate(
            [prompt],
            diff_matrices=char_differences, 
            max_new_tokens=1
        )[0]
        # Clean
        cleaned_answer = cleaning(generated_output)

        # Restoer answers
        answer_key = f"answer_{character.replace(' ', '_')}"
        sample[answer_key] = cleaned_answer
        accuracy_counts[character]["total"] += 1  # Included in total

        # Determine the answer
        if cleaned_answer in ["A", "B", "C", "D"]:
            if cleaned_answer == true_label:
                accuracy_counts[character]["correct"] += 1
        elif cleaned_answer == "E":
            # Does not count the accuracy, only records the number of E times
            accuracy_counts[character]["E_count"] += 1
        else:
            # Not within the range of ABCDE -> Treat as invalid
            # Try calling handle_invalid_answer again to see if the output can be corrected
            corrected_answer, is_correct = handle_invalid_answer(
                vc=vc,
                prompt=prompt,
                true_label_text=true_label_text,
                true_label=true_label,
                diff_matrices=char_differences,
                max_new_tokens=8
            )
            # Write the final answer back
            sample[answer_key] = corrected_answer.upper()
            
            if is_correct:
                accuracy_counts[character]["correct"] += 1
            else:
                accuracy_counts[character]["invalid"] += 1

# Calculate accuracy
accuracy_results = {}
for character in characters:
    counts = accuracy_counts[character]
    correct = counts["correct"]
    total = counts["total"]
    E_count = counts["E_count"]
    invalid = counts["invalid"]
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    accuracy_results[character] = {
        "correct": correct,
        "total": total,
        "E_count": E_count,
        "invalid": invalid,
        "accuracy_percentage": round(accuracy, 2)
    }

    print(f"Accuracy for {character}: {accuracy_results[character]['accuracy_percentage']}% "
          f"({correct}/{total})")
    print(f"Number of 'E' answers for {character}: {E_count}")
    print(f"Number of invalid answers for {character}: {invalid}")

# Restore JSON
final_output = {
    "data": data,
    "accuracy": accuracy_results
}

answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers.json")
print("Saving generated answers and accuracy to JSON...")
with open(answers_save_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)
print(f"Saved answers and accuracy to {answers_save_path}")
print("All answers and accuracy have been saved successfully.")