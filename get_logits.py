#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 22:20:52 2025
Modified    May  8 2025 to embed TASKS list

This script extracts the highest-logit option (and its probability) for each role
across multiple tasks defined in TASKS, saving results per role into CSV files.
"""

import os
import json
import numpy as np
import csv
from vicuna import VicundaModel
from tqdm import tqdm  # progress bar
import torch
from detection.task_list import TASKS


# ------------------------- Configuration -------------------------
# List of tasks to process
# Model settings
MODEL_NAME = "llama3"
SIZE = "8B"
MODEL_DIR = "meta-llama/Llama-3.1-8B-Instruct"
NUM_GPUS = 3

# File paths
MMLU_DIR = "/data2/paveen/RolePlaying/src/models/components/mmlu"
SAVE_DIR = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
os.makedirs(SAVE_DIR, exist_ok=True)

# Label mapping for multiple-choice
LABEL_MAPPING = ["A", "B", "C", "D"]

def make_characters(task_name: str):
    task_name = task_name.replace("_", " ")
    return [f"non {task_name} expert",   # ← add f
            f"{task_name} student",
            f"{task_name} expert",
            "person"]
# ------------------------- Helper Functions ------------------------

def load_json_data(json_path: str):
    print(f"Loading JSON data from {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")
    return data


def get_option_token_ids(vc: VicundaModel):
    """
    Map options "A","B","C","D" to their single-token IDs.
    """
    ids = []
    for opt in LABEL_MAPPING:
        token_ids = vc.tokenizer.encode(opt, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option '{opt}' maps to tokens {token_ids}, expected exactly one.")
        ids.append(token_ids[0])
    return ids


def compute_softmax(logits: np.ndarray):
    """Compute softmax over a 1D array of logits."""
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()


def get_clean_role(role: str):
    """Convert a role string to a short, filesystem-safe name."""
    tokens = role.split()
    if role.lower() == "person":
        return "person"
    if tokens[0].lower() == "none":
        return f"none_{tokens[-1].lower()}"
    return tokens[-1].lower()

# ------------------------------ Main -------------------------------
def main():
    # 1) Initialize model once
    print(f"Loading model from {MODEL_DIR} ...")
    vc = VicundaModel(model_path=MODEL_DIR, num_gpus=1)
    vc.model.eval()
    print(vc.template)
    option_token_ids = get_option_token_ids(vc)

    # 2) Roles to test (each script run will use these prompts)
    for task in TASKS:
        print(f"\n--- Processing task: {task} ---")
        print(vc.template)
        data_path = os.path.join(MMLU_DIR, f"{task}.json")
        data = load_json_data(data_path)

        roles = make_characters(task)

        # Prepare storage for this task
        role_summary = {role: {"logits": [], "probs": []} for role in roles}
        # NEW: count correct predictions per role
        role_correct = {role: 0 for role in roles}

        # 3) Iterate samples
        with torch.no_grad():
            for sample in tqdm(data, desc=f"{task}", unit="sample"):
                context   = sample.get("text", "")
                label_int = sample.get("label", -1)
                if not (0 <= label_int < len(LABEL_MAPPING)):
                    continue
                true_label = LABEL_MAPPING[label_int]

                for role in roles:
                    prompt        = vc.template.format(character=role, context=context)
                    logits_tensor = vc.get_logits([prompt], character=role)
                    last_logits   = logits_tensor[0, -1, :].detach().cpu().numpy()
                    option_logits = np.array([last_logits[t] for t in option_token_ids])
                    option_probs  = compute_softmax(option_logits)

                    pred_idx   = int(np.argmax(option_logits))
                    pred_label = LABEL_MAPPING[pred_idx]

                    # NEW: increment correct count if prediction matches true_label
                    if pred_label == true_label:
                        role_correct[role] += 1

                    # Record the highest-logit option regardless of correctness
                    role_summary[role]["logits"].append(option_logits[pred_idx])
                    role_summary[role]["probs"].append(option_probs[pred_idx])
        
        # 4) Compute and save metrics
        torch.cuda.empty_cache()
        print(f"\nResults for task '{task}':")
        for role in roles:
            values   = role_summary[role]
            total    = len(values["logits"])          
            correct  = role_correct[role]             
            acc_pct  = (correct / total * 100) if total else 0.0

            avg_logit = float(np.mean(values["logits"])) if values["logits"] else None
            avg_prob  = float(np.mean(values["probs"]))  if values["probs"]  else None

            print(
                f"Role={role:<20}  Total={total:>5}  "
                f"Correct={correct:>5}  Acc={acc_pct:5.2f}%  "
                f"AvgLogit={avg_logit!s:<8}  AvgProb={avg_prob!s}"
            )

            # Append to CSV
            clean    = get_clean_role(role)
            csv_path = os.path.join(SAVE_DIR, f"{clean}.csv")
            exists   = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["Task", "Total", "Correct", "Accuracy(%)", "Avg Logit", "Avg Prob"])
                writer.writerow([task, total, correct, round(acc_pct, 2), avg_logit, avg_prob])
            print(f"[CSV] Appended to {csv_path}")

if __name__ == "__main__":
    main()
