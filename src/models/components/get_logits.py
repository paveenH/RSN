#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 22:20:52 2025

@author: paveenhuang
"""

import argparse
import os
import json
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
    
    # Define the list of roles based on the task (for prompt generation, role includes task)
    roles = [f"none {task} expert", f"{task} student", f"{task} expert", "person"]
    return task, model, size, roles

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

def get_clean_role(role):
    """
    Convert the role string to a "clean" name for file saving.
    For example:
      "none anatomy expert" -> "none_expert"
      "anatomy student" -> "student"
      "anatomy expert" -> "expert"
      "person" -> "person"
    """
    tokens = role.split()
    if role.lower() == "person":
        return "person"
    elif tokens[0].lower() == "none":
        return "none_" + tokens[-1].lower()
    else:
        return tokens[-1].lower()

def main():
    # 1) Get parameters and roles
    task, model_name, size, roles = parse_arguments_and_define_characters()
    
    # 2) Define paths
    data_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
    os.makedirs(save_dir, exist_ok=True)
    
    # 3) Load model
    vc = VicundaModel(model_path=model_path, num_gpus=4)
    
    # 4) Load task data
    data = load_json_data(data_path)
    
    # 5) Get token IDs for options "A", "B", "C", "D"
    option_token_ids = get_option_token_ids(vc)
    
    # 6) For each role, store correct sample results (logits and probs)
    role_summary = {role: {"logits": [], "probs": []} for role in roles}
    
    # 7) Iterate through the dataset
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        label_int = sample.get("label", -1)
        # Skip invalid label
        if not (0 <= label_int < len(LABEL_MAPPING)):
            continue
        true_label = LABEL_MAPPING[label_int]
        
        for role in roles:
            # Use the full role (with task) for constructing prompt
            prompt = vc.template.format(character=role, context=context)
            logits = vc.get_logits([prompt], character=role)  # shape: [1, seq_len, vocab_size]
            last_logits = logits[0, -1, :].detach().cpu().numpy()
            option_logits = np.array([last_logits[tid] for tid in option_token_ids])
            option_probs = compute_softmax(option_logits)
            
            pred_idx = int(np.argmax(option_probs))
            pred_label = LABEL_MAPPING[pred_idx]
            
            if pred_label == true_label:
                correct_logit = option_logits[label_int]
                correct_prob  = option_probs[label_int]
                role_summary[role]["logits"].append(correct_logit)
                role_summary[role]["probs"].append(correct_prob)
    
    # 8) Compute metrics for each role: sample count, average logit, average probability
    role_metrics = {}
    for role in roles:
        n_correct = len(role_summary[role]["logits"])
        if n_correct > 0:
            avg_logit = float(np.mean(role_summary[role]["logits"]))
            avg_prob  = float(np.mean(role_summary[role]["probs"]))
        else:
            avg_logit = None
            avg_prob  = None
        role_metrics[role] = {
            "n": n_correct,
            "avg_logit": avg_logit,
            "avg_prob": avg_prob
        }
    
    # 9) Print summary
    print(f"--- Task={task} Summary ---")
    for role in roles:
        print(f"Role: {role}, Samples: {role_metrics[role]['n']}, "
              f"Avg Logit: {role_metrics[role]['avg_logit']}, "
              f"Avg Prob: {role_metrics[role]['avg_prob']}")
    
    # 10) For each role, append one row to its corresponding CSV file.
    #     CSV header: ["Task", "Samples", "Avg Logit", "Avg Prob"]
    for role in roles:
        clean_role = get_clean_role(role)
        csv_path = os.path.join(save_dir, f"{clean_role}.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Task", "Samples", "Avg Logit", "Avg Prob"])
            row = [task,
                   role_metrics[role]["n"],
                   role_metrics[role]["avg_logit"],
                   role_metrics[role]["avg_prob"]]
            writer.writerow(row)
        print(f"[Saved CSV] {csv_path} for role: {clean_role}, task: {task}")

if __name__ == "__main__":
    main()