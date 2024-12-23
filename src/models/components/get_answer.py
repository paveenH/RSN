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

# We only focus on generating answers here, not hidden states
template = "Would you answer the following question with A, B, C or D? \nQuestion: {context}\nAnswer A/B/C/D as a {character}:"
# template = "You are a {character}, You are a {character}, You are a {character}, would you answer the following question with A, B, C or D? \n Question: {context}\n Answer: "

characters = ["management expert", "medical genetics expert"]

# Load JSON data
print(f"Loading JSON data from {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples loaded: {len(data)}")

# Initialize storage for generated answers
generated_answers_storage = {character: [] for character in characters}

print("Starting answer generation...")
for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
    context = sample.get("text", "")
    if not context:
        print(f"Sample {idx} is missing 'text' field. Skipping.")
        continue

    for character in characters:
        # Generate prompt
        prompt = template.format(character=character, context=context)
        # Generate answer using vc.generate
        # vc.generate expects a list of prompts and returns a list of answers
        generated_output = vc.generate([prompt])[0]  # Get the single output
        
        # Store this raw answer
        generated_answers_storage[character].append(generated_output.strip())

print("Finished generating answers.")

# Save generated answers to JSON
answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers.json")
print("Saving generated answers to JSON...")
with open(answers_save_path, "w", encoding="utf-8") as f:
    json.dump(generated_answers_storage, f, ensure_ascii=False, indent=4)
print(f"Saved generated answers for description to {answers_save_path}")

print("All answers have been saved successfully.")