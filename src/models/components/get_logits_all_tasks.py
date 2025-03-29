#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 22:20:52 2025

@author: paveenhuang
"""

import os
import json
import numpy as np
import csv
from vicuna import VicundaModel

# Label mapping: A, B, C, D
LABEL_MAPPING = ["A", "B", "C", "D"]

# Define all tasks in a global list
TASKS = [
    # "abstract_algebra",
    # "anatomy",
    # "astronomy",
    # "business_ethics",
    # "clinical_knowledge",
    # "college_biology",
    # "college_chemistry",
    # "college_computer_science",
    "college_medicine",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

model_name = "llama3"
size = "8B"

def load_json_data(json_path):
    """Load the MMLU data from JSON"""
    if not os.path.exists(json_path):
        print(f"[Warning] {json_path} does not exist. Skipping.")
        return None
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_option_token_ids(vc):
    """Get the token ids for 'A','B','C','D'. Each is assumed to be exactly 1 token."""
    option_token_ids = []
    for option in LABEL_MAPPING:
        token_ids = vc.tokenizer.encode(option, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option {option} does not map to exactly one token: {token_ids}")
        option_token_ids.append(token_ids[0])
    return option_token_ids

def compute_softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def get_clean_role(role):
    """
    Convert the role string to a short name for CSV.
      e.g. "none anatomy expert" -> "none_expert"
           "anatomy student"     -> "student"
           "anatomy expert"      -> "expert"
           "person"              -> "person"
    """
    tokens = role.split()
    if role.lower() == "person":
        return "person"
    elif tokens[0].lower() == "none":
        return "none_" + tokens[-1].lower()
    else:
        return tokens[-1].lower()

def run_one_task(task, vc, option_token_ids, save_dir):
    """
    Process one task with the given model and token_ids,
    then append results to CSV for each role.
    """

    # MMLU data path
    data_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    data = load_json_data(data_path)
    if data is None:
        return  # skip if file doesn't exist or can't load

    print(f"[Task: {task}] Loaded {len(data)} samples.")

    # Roles for this task
    roles = [f"none {task} expert", f"{task} student", f"{task} expert", "person"]

    # For each role, collect correct predictions
    role_summary = {role: {"logits": [], "probs": []} for role in roles}

    # Iterate samples
    for sample in data:
        label_int = sample.get("label", -1)
        if not (0 <= label_int < len(LABEL_MAPPING)):
            continue
        true_label = LABEL_MAPPING[label_int]
        context = sample.get("text", "")

        for role in roles:
            prompt = vc.template.format(character=role, context=context)
            logits = vc.get_logits([prompt], character=role)  # [1, seq_len, vocab_size]
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

    # Summarize: sample count, average logit, average probability
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

    # Print summary
    print(f"--- Task={task} Summary ---")
    for role in roles:
        print(f"Role: {role}, Samples={role_metrics[role]['n']}, "
              f"Avg Logit={role_metrics[role]['avg_logit']}, "
              f"Avg Prob={role_metrics[role]['avg_prob']}")

    # Save CSV per role
    for role in roles:
        clean_role = get_clean_role(role)
        csv_path = os.path.join(save_dir, f"{clean_role}.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Task", "Samples", "Avg Logit", "Avg Prob"])
            row = [
                task,
                role_metrics[role]["n"],
                role_metrics[role]["avg_logit"],
                role_metrics[role]["avg_prob"]
            ]
            writer.writerow(row)
        print(f"[Saved CSV] {csv_path} for role: {clean_role}, task: {task}")

def main():    
    # 1) Load model once
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    vc = VicundaModel(model_path=model_path, num_gpus=7)
    
    # 2) Prepare for saving
    save_dir = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
    os.makedirs(save_dir, exist_ok=True)

    # 3) Get ABCD token IDs
    option_token_ids = get_option_token_ids(vc)

    # 4) Loop all tasks
    for task in TASKS:
        run_one_task(task, vc, option_token_ids, save_dir)

    print("All tasks processed. Model was loaded only once!")

if __name__ == "__main__":
    main()