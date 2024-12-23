#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:43:55 2024

@author: paveenhuang
"""

import json
import argparse
import os
from tqdm import tqdm

from vicuna import VicundaModel

# Define the path to the JSON files
PATH = "/data2/paveen/RolePlaying/src/models/components/mmlu"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
parser.add_argument("task", type=str, help="The name of the task to process.")
parser.add_argument("size", type=str, help="The size of the model (e.g., 13B).")
args = parser.parse_args()

task = args.task
size = args.size

# Define model path
model_path = f"/data2/paveen/RolePlaying/shared/llama3/{size}"

# Define JSON file path
json_path = os.path.join(PATH, f"{task}.json")

# Define save directory for answers
save_dir = os.path.join("/data2/paveen/RolePlaying/src/models/components/answer")
os.makedirs(save_dir, exist_ok=True)

# Initialize VicundaModel
vc = VicundaModel(model_path=model_path)

# Define the template for generating answers
template = "Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you thinking as a {character}, your answer is:"
# template = "Would you answer the following question with A, B, C or D?\nQuestion: {context}\nNow suppose you are a {character}, your answer is:"
# template = "You are a {character}, You are a {character}, You are a {character}, would you answer the following question with A, B, C or D? \n Question: {context}\n Answer: "
# template = "You are a {character}, would you answer the following question with A, B, C or D? \n Question: {context}\n Answer: "

# Define the list of characters
characters = ["management expert", "medical genetics expert"]

# Load JSON data
print(f"Loading JSON data from {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples loaded: {len(data)}")

# Initialize storage for generated answers
generated_answers_storage = {character: [] for character in characters}

# Initialize accuracy tracking
accuracy_counts = {character: {"correct": 0, "total": 0} for character in characters}
label_mapping = ["A", "B", "C", "D"]

print("Starting answer generation and accuracy calculation...")
for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
    context = sample.get("text", "")
    true_label_int = sample.get("label", -1)   # Ensure label is uppercase and stripped
    true_label = label_mapping[true_label_int]
    if not context:
        print(f"Sample {idx} is missing 'text' field. Skipping.")
        continue

    for character in characters:
        # Generate prompt
        prompt = template.format(character=character, context=context)

        # Generate answer using vc.generate
        generated_output = vc.generate([prompt])[0]  # Get the single output

        # Clean and parse the generated answer
        generated_answer = generated_output.strip().upper()
        if generated_answer not in ["A", "B", "C", "D", "E"]:
            # If the generated answer is invalid, assign a default value
            default_answer = "E"
            print(f"Sample {idx}, Character '{character}': Invalid generated answer '{generated_answer}'. Defaulted to '{default_answer}'.")
            # generated_answer = default_answer

        # Add the generated answer to the sample
        answer_key = f"answer_{character.replace(' ', '_')}"
        sample[answer_key] = generated_answer

        # Update generated answers storage
        generated_answers_storage[character].append(generated_answer)

        # Update accuracy counts
        if generated_answer == true_label:
            accuracy_counts[character]["correct"] += 1
        accuracy_counts[character]["total"] += 1

# After processing all samples, compute accuracy
accuracy_results = {}
for character in characters:
    correct = accuracy_counts[character]["correct"]
    total = accuracy_counts[character]["total"]
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    accuracy_results[character] = {
        "correct": correct,
        "total": total,
        "accuracy_percentage": round(accuracy, 2)
    }
    print(f"Accuracy for {character}: {accuracy_results[character]['accuracy_percentage']}% ({correct}/{total})")

# Prepare the final JSON structure
final_output = {
    "data": data,
    "accuracy": accuracy_results
}

# Save the modified data and accuracy to JSON
answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers.json")
print("Saving generated answers and accuracy to JSON...")
with open(answers_save_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)
print(f"Saved answers and accuracy to {answers_save_path}")

print("All answers and accuracy have been saved successfully.")