#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 22:20:52 2025

@author: paveenhuang
"""

import argparse
import json
import os
import numpy as np
from vicuna import VicundaModel

# Label mapping: A, B, C, D correspond to index 0,1,2,3
LABEL_MAPPING = ["A", "B", "C", "D"]

def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

def parse_arguments_and_define_characters():
    """
    Parse command-line arguments, return task, model, size
    and define the list of characters to be tested.
    """
    parser = argparse.ArgumentParser(description="Extract logits for each role")
    parser.add_argument("task_size", type=str, help="The task, model, and size as a combined argument.")
    args = parser.parse_args()
    try:
        task, model, size = args.task_size.split()
    except ValueError:
        raise ValueError("The task_size parameter should contain three parts: task, model, and size.")
    
    # Define the list of characters (modify based on your requirements)
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
    Use the model's tokenizer to get the token IDs corresponding to the options "A", "B", "C", "D" 
    (assuming each option is a single token).
    """
    option_token_ids = []
    for option in LABEL_MAPPING:
        # Note: This assumes each option encodes to exactly one token
        token_ids = vc.tokenizer.encode(option, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option {option} does not map to exactly one token: {token_ids}")
        option_token_ids.append(token_ids[0])
    return option_token_ids

def compute_softmax(logits):
    """Compute the softmax to obtain the probability distribution"""
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def main():
    # Parse arguments
    task, model_name, size, roles = parse_arguments_and_define_characters()
    
    # Define paths
    data_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the model, recommend multi-GPU loading (adjust based on your environment, like using CUDA_VISIBLE_DEVICES=...)
    vc = VicundaModel(model_path=model_path, num_gpus=2)
    
    # Load data
    data = load_json_data(data_path)
    
    # Get token IDs for options "A", "B", "C", "D"
    option_token_ids = get_option_token_ids(vc)
    
    # Initialize a dictionary to store the correct predictions and corresponding logits/probabilities for each role
    results = {role: [] for role in roles}
    
    # Iterate through the dataset (you can also limit the number of samples here)
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        true_label_int = sample.get("label", -1)
        true_label = LABEL_MAPPING[true_label_int]  # For example, "C"
        
        # For each role, construct the corresponding prompt and get the logits
        for role in roles:
            # Construct the prompt using the model's template
            prompt = vc.template.format(character=role, context=context)
            logits = vc.get_logits([prompt], character=role)  # logits: tensor, shape (1, seq_len, vocab_size)
            last_logits = logits[0, -1, :].detach().cpu().numpy()
            
            # Extract the logits corresponding to options "A", "B", "C", "D"
            option_logits = [last_logits[tid] for tid in option_token_ids]
            
            # Get the softmax probabilities (or directly compare the raw logits)
            option_probs = compute_softmax(np.array(option_logits))
            
            # The predicted option is the one with the highest probability
            pred_idx = int(np.argmax(option_probs))
            pred_label = LABEL_MAPPING[pred_idx]
            
            # If the prediction is correct, save the correct option's logit and probability
            if pred_label == true_label:
                # Save a set of data, including the original logit and softmax probabilities
                results[role].append({
                    "sample_idx": idx,
                    "true_label": true_label,
                    "predicted_logit": option_logits[true_label_int],
                    "predicted_prob": option_probs[true_label_int],
                    "all_option_logits": option_logits,
                    "all_option_probs": option_probs,
                })
    
    # Output results: calculate the average logit and probability for the correct predictions per role
    summary = {}
    for role, vals in results.items():
        if vals:
            avg_logit = np.mean([v["predicted_logit"] for v in vals])
            avg_prob = np.mean([v["predicted_prob"] for v in vals])
            summary[role] = {"avg_logit": float(avg_logit), "avg_prob": float(avg_prob), "n": len(vals)}
        else:
            summary[role] = {"avg_logit": None, "avg_prob": None, "n": 0}
    
    print("Summary of logits on correctly predicted samples per role:")
    for role, info in summary.items():
        print(f"Role: {role}  Samples: {info['n']}  Avg Logit: {info['avg_logit']}, Avg Prob: {info['avg_prob']}")
    
    # Save the detailed results and summary to a JSON file
    output = {"detailed": results, "summary": summary}
    output_serializable = convert_numpy_types(output)
    out_path = os.path.join(save_dir, f"logits_{task}_{model_name}_{size}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_serializable, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()