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

# ------------------------- Configuration -------------------------
# List of tasks to process
TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
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
    "world_religions"
]
# Model settings
MODEL_NAME = "llama3"
SIZE = "8B"
NUM_GPUS = 1

# File paths
MMLU_DIR = "/data2/paveen/RolePlaying/src/models/components/mmlu"
SHARED_MODEL_DIR = "/data2/paveen/RolePlaying/shared"
SAVE_DIR = "/data2/paveen/RolePlaying/src/models/components/logits_v3_4ops"
os.makedirs(SAVE_DIR, exist_ok=True)

# Label mapping for multiple-choice
LABEL_MAPPING = ["A", "B", "C", "D"]

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
    model_path = os.path.join(SHARED_MODEL_DIR, MODEL_NAME, SIZE)
    print(f"Loading model from {model_path} ...")
    vc = VicundaModel(model_path=model_path, num_gpus=2)
    option_token_ids = get_option_token_ids(vc)

    # 2) Roles to test (each script run will use these prompts)
    # 'none <task> expert', '<task> student', '<task> expert', 'person'
    for task in TASKS:
        print(f"\n--- Processing task: {task} ---")
        data_path = os.path.join(MMLU_DIR, f"{task}.json")
        data = load_json_data(data_path)

        roles = [
            f"none {task} expert",
            f"{task} student",
            f"{task} expert",
            "person"
        ]

        # Prepare storage for this task
        role_summary = {role: {"logits": [], "probs": []} for role in roles}

        # 3) Iterate samples
        for sample in tqdm(data, desc=f"{task}", unit="sample"):
            context = sample.get("text", "")
            label_int = sample.get("label", -1)
            if not (0 <= label_int < len(LABEL_MAPPING)):
                continue

            for role in roles:
                prompt = vc.template.format(character=role, context=context)
                logits_tensor = vc.get_logits([prompt], character=role)
                last_logits = logits_tensor[0, -1, :].detach().cpu().numpy()
                option_logits = np.array([last_logits[t] for t in option_token_ids])
                option_probs = compute_softmax(option_logits)

                # Record the highest-logit option
                pred_idx = int(np.argmax(option_logits))
                pred_logit = option_logits[pred_idx]
                pred_prob = option_probs[pred_idx]

                role_summary[role]["logits"].append(pred_logit)
                role_summary[role]["probs"].append(pred_prob)

        # 4) Compute and save metrics
        print(f"\nResults for task '{task}':")
        for role in roles:
            values = role_summary[role]
            n_samples = len(values["logits"])
            avg_logit = float(np.mean(values["logits"])) if n_samples > 0 else None
            avg_prob = float(np.mean(values["probs"])) if n_samples > 0 else None
            print(f"Role={role:<20}  Samples={n_samples:>5}  AvgLogit={avg_logit!s:<8}  AvgProb={avg_prob!s}")

            # Append to CSV
            clean = get_clean_role(role)
            csv_path = os.path.join(SAVE_DIR, f"{clean}.csv")
            exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["Task", "Samples", "Avg Logit", "Avg Prob"])
                writer.writerow([task, n_samples, avg_logit, avg_prob])
            print(f"[CSV] Appended to {csv_path}")

    print("\nAll tasks processed.")

if __name__ == "__main__":
    main()
