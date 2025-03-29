#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 22:20:52 2025

@author: paveenhuang
"""

import argparse
import os
import numpy as np
import csv  # Used for saving CSV
from vicuna import VicundaModel

# Label mapping: A, B, C, D
LABEL_MAPPING = ["A", "B", "C", "D"]

def parse_arguments_and_define_characters():
    """
    Parse command-line arguments and return task, model, size
    and define the list of roles to be tested.
    """
    parser = argparse.ArgumentParser(description="Extract logits for each role")
    parser.add_argument("task_size", type=str, help="The task, model, and size as a combined argument.")
    args = parser.parse_args()
    try:
        task, model, size = args.task_size.split()
    except ValueError:
        raise ValueError("The task_size parameter should contain three parts: task, model, and size.")
    
    # Define the list of roles based on the task (this is based on your example)
    characters = [f"none {task} expert", f"{task} student", f"{task} expert", "person"]
    return task, model, size, characters

def load_json_data(json_path):
    print(f"Loading JSON data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total samples loaded: {len(data)}")
    return data

def get_option_token_ids(vc):
    """
    Get the token ids corresponding to options "A", "B", "C", "D"
    (assuming each option corresponds to 1 token).
    """
    option_token_ids = []
    for option in LABEL_MAPPING:
        token_ids = vc.tokenizer.encode(option, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option {option} does not map to exactly one token: {token_ids}")
        option_token_ids.append(token_ids[0])
    return option_token_ids

def compute_softmax(logits):
    """Compute the softmax of logits to obtain the probability distribution"""
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def main():
    # 1) Get parameters and roles
    task, model_name, size, roles = parse_arguments_and_define_characters()
    
    # 2) Define paths
    data_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
    os.makedirs(save_dir, exist_ok=True)
    
    # 3) Load model
    vc = VicundaModel(model_path=model_path, num_gpus=2)
    
    # 4) Load the task data
    data = load_json_data(data_path)
    
    # 5) Get token IDs for "A", "B", "C", "D"
    option_token_ids = get_option_token_ids(vc)
    
    # 6) For each role, store the logits and probabilities for correctly predicted samples
    #    Only store the data we want to output to CSV (avg_logit, avg_prob, n)
    role_summary = {role: {"logits": [], "probs": []} for role in roles}
    
    # 7) Iterate through the dataset
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        label_int = sample.get("label", -1)
        
        # Skip if label_int is invalid
        if not (0 <= label_int < len(LABEL_MAPPING)):
            continue
        true_label = LABEL_MAPPING[label_int]
        
        # Run for each role
        for role in roles:
            prompt = vc.template.format(character=role, context=context)
            
            # logits: [1, seq_len, vocab_size]
            logits = vc.get_logits([prompt], character=role)
            
            # Get the logits of the last token => shape [vocab_size]
            last_logits = logits[0, -1, :].detach().cpu().numpy()
            # Extract logits for "A", "B", "C", "D"
            option_logits = np.array([last_logits[tid] for tid in option_token_ids])
            # softmax
            option_probs = compute_softmax(option_logits)
            
            # Predict the option
            pred_idx = int(np.argmax(option_probs))
            pred_label = LABEL_MAPPING[pred_idx]
            
            # If the prediction is correct
            if pred_label == true_label:
                # Record the correct option's logit / prob
                correct_logit = option_logits[label_int]
                correct_prob  = option_probs[label_int]
                
                role_summary[role]["logits"].append(correct_logit)
                role_summary[role]["probs"].append(correct_prob)
    
    # 8) Calculate the average for each role
    #    Generate columns to write to the CSV: [role1_avg_logit, role2_avg_logit, ...]
    #    You can also add avg_prob and others
    role_avg_logits = {}
    for role in roles:
        if len(role_summary[role]["logits"]) > 0:
            avg_logit = float(np.mean(role_summary[role]["logits"]))
            # avg_prob  = float(np.mean(role_summary[role]["probs"]))
        else:
            avg_logit = None
            # avg_prob = None
        role_avg_logits[role] = avg_logit
    
    # 9) Output information
    print(f"--- Task={task} Summary (avg_logit) ---")
    for role in roles:
        print(f"Role: {role}, avg_logit={role_avg_logits[role]}")
    
    # 10) Write to CSV
    #     Row: task
    #     Columns: role1, role2, role3, ...
    # Example: "Task, none_{task}_expert, {task}_student, {task}_expert, person"
    
    csv_path = os.path.join(save_dir, "logits_summary.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # If the file doesn't exist, write the header
        if not file_exists:
            header = ["Task"] + roles
            writer.writerow(header)
        
        # Write the average logits for each role
        row = [task] + [role_avg_logits[r] for r in roles]
        writer.writerow(row)
    
    print(f"[Saved CSV] {csv_path} for task={task}")

if __name__ == "__main__":
    main()