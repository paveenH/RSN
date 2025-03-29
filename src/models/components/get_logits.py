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
    # 1) Parse arguments
    task, model_name, size, roles = parse_arguments_and_define_characters()
    
    # 2) Define paths
    data_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
    os.makedirs(save_dir, exist_ok=True)
    
    # 3) Initialize model
    vc = VicundaModel(model_path=model_path, num_gpus=2)
    
    # 4) Load data
    data = load_json_data(data_path)
    
    # 5) Get token IDs for options A,B,C,D
    option_token_ids = get_option_token_ids(vc)
    
    # 6) Prepare result structure
    results = {role: [] for role in roles}
    
    # 7) Iterate through samples
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        true_label_int = sample.get("label", -1)
        true_label = LABEL_MAPPING[true_label_int]  # e.g. "C"
        
        # For each role, build prompt and compute logits
        for role in roles:
            prompt = vc.template.format(character=role, context=context)
            logits = vc.get_logits([prompt], character=role)
            
            last_logits = logits[0, -1, :].detach().cpu().numpy()
            option_logits = [last_logits[tid] for tid in option_token_ids]
            option_probs = compute_softmax(np.array(option_logits))
            
            pred_idx = int(np.argmax(option_probs))
            pred_label = LABEL_MAPPING[pred_idx]
            
            if pred_label == true_label:
                results[role].append({
                    "sample_idx": idx,
                    "true_label": true_label,
                    "predicted_logit": option_logits[true_label_int],
                    "predicted_prob": option_probs[true_label_int],
                    "all_option_logits": option_logits,
                    "all_option_probs": option_probs,
                })
    
    # 8) Summarize
    summary = {}
    for role, vals in results.items():
        if vals:
            avg_logit = np.mean([v["predicted_logit"] for v in vals])
            avg_prob = np.mean([v["predicted_prob"] for v in vals])
            summary[role] = {
                "avg_logit": float(avg_logit),
                "avg_prob": float(avg_prob),
                "n": len(vals)
            }
        else:
            summary[role] = {"avg_logit": None, "avg_prob": None, "n": 0}
    
    print("Summary of logits on correctly predicted samples per role:")
    for role, info in summary.items():
        print(
            f"Role: {role}  Samples: {info['n']}  "
            f"Avg Logit: {info['avg_logit']}, Avg Prob: {info['avg_prob']}"
        )
    
    output = {
        "detailed": results,
        "summary": summary
    }
    
    # 9) Convert and save
    output_serializable = convert_numpy_types(output)
    out_path = os.path.join(save_dir, f"logits_{task}_{model_name}_{size}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_serializable, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()