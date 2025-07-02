#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 22:20:52 2025
Modified    May  8 2025 to embed TASKS list

This script extracts the highest-logit option (and its probability) for each role
across multiple tasks defined in TASKS, saving results per role into CSV files.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm  # progress bar

import get_answer_alltasks as ga
from llms import VicundaModel

# ------------------------- Configuration -------------------------
# List of tasks to process
TASKS = ga.TASKS
MODEL_NAME = "mistral"
SIZE = "7B"
TYPE = "non"

LABEL_MAPPING = ["A", "B", "C", "D", "E"]

MMLU_DIR = Path("/data2/paveen/RolePlaying/components/mmlu")
MODEL_DIR = "mistralai/Mistral-7B-v0.3"
print("Loading model from: ", MODEL_DIR)
SAVE_BASE = Path(f"/data2/paveen/RolePlaying/components/answer_softmax_{TYPE}")
SAVE_BASE.mkdir(parents=True, exist_ok=True)

# ------------------------- Helper Functions ------------------------
def get_option_token_ids(vc: VicundaModel):
    """
    Map options "A","B","C","D" to their single-token IDs.
    """
    ids = []
    for opt in LABEL_MAPPING:
        token_ids = vc.tokenizer(opt, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Option '{opt}' maps to tokens {token_ids}, expected exactly one.")
        ids.append(token_ids[0])
    return ids


def compute_softmax(logits: np.ndarray):
    """Compute softmax over a 1D array of logits."""
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def role_key(role: str, suffix: str) -> str:
    """Generate JSON key: spaces→underscore then add suffix."""
    return f"{role.replace(' ', '_')}_{suffix}"

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ------------------------------ Main -------------------------------
def main():
    vc = VicundaModel(model_path=MODEL_DIR)
    vc.model.eval()
    option_token_ids = get_option_token_ids(vc)

    for task in TASKS:
        print("\n===", task, "===")
        print("Template: ", vc.template)
        data_path = MMLU_DIR / f"{task}.json"
        if not data_path.exists():
            print("[Skip]", data_path, "not found")
            continue

        samples = ga.load_json(data_path)
        roles    = ga.make_characters(task, TYPE)

        # per‑role accuracy counters
        role_stats: Dict[str, Dict[str, int]] = {
            r: {"correct": 0, "E": 0, "invalid": 0, "total": 0} for r in roles
        }

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                context     = sample["text"]
                true_idx    = sample["label"]
                if not 0 <= true_idx < len(LABEL_MAPPING):
                    continue
                true_label  = LABEL_MAPPING[true_idx]

                for role in roles:
                    prompt = vc.template.format(character=role, context=context)
                    logits = vc.get_logits([prompt])[0, -1].cpu().numpy()
                    option_logits = np.array([logits[t] for t in option_token_ids])
                    probs         = compute_softmax(option_logits)

                    pred_idx   = int(option_logits.argmax())
                    pred_label = LABEL_MAPPING[pred_idx]
                    pred_prob  = float(probs[pred_idx])

                    # record answers + prob in sample dict
                    sample[role_key(role, "answer")] = pred_label
                    sample[role_key(role, "prob")]   = pred_prob

                    # statistics
                    role_stats[role]["total"] += 1
                    if pred_label == true_label:
                        role_stats[role]["correct"] += 1
                    elif pred_label == "E":
                        role_stats[role]["E"] += 1
                    else:
                        role_stats[role]["invalid"] += 1

        # build accuracy summary like old format
        accuracy = {}
        for role, s in role_stats.items():
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0.0
            accuracy[role] = {
                **s,
                "accuracy_percentage": round(pct, 2),
            }
            print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']})")

        # save
        out_dir  = SAVE_BASE / MODEL_DIR.split("/")[-1]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{task}_{TYPE}.json"
        save_json({"data": samples, "accuracy": accuracy}, out_file)
        print("[Saved]", out_file)

    print("\n✅  All tasks finished.")

if __name__ == "__main__":
    main()
