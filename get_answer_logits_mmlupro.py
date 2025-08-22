#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 17:27:15 2025

Extract highest-logit answer + probability **and** save last-token hidden states
for every role on every task — switched to MMLU-Pro combined JSON.

@author: paveenhuang

"""

import json
from pathlib import Path
from typing import List
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv

from llms import VicundaModel
from template import select_templates
from utils import load_json, make_characters, option_token_ids, construct_prompt


# ───────────────────── Helper functions ─────────────────────────

def softmax_1d(x: np.ndarray):
    e = np.exp(x - x.max())
    return e / e.sum()

def rkey(role: str, suf: str):
    return f"{suf}_{role.replace(' ', '_')}"

def dump_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def letters_k(k: int) -> List[str]:
    k = int(k)
    k = max(1, min(10, k))
    return [chr(ord("A") + i) for i in range(k)]


# ─────────────────────────── Main ───────────────────────────────

def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load mmlupro json file
    all_samples: List[dict] = load_json(Path(args.mmlupro_json))

    # group by "task"
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} tasks in MMLU-Pro JSON.")

    templates = select_templates(args.use_E) 

    for task in tasks:
        print(f"\n=== {task} ===")
        samples = [s for s in all_samples if s["task"] == task]
        if not samples:
            print("[Skip] empty task:", task)
            continue

        # number of options in task
        max_label = max(int(s["label"]) for s in samples)
        K = max_label + 1
        K = max(1, min(10, K))
        LABELS = letters_k(K)

        # get ids of options
        opt_ids = option_token_ids(vc, LABELS)

        # role list
        roles = make_characters(task.replace(" ", "_"), args.type)
        role_stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

        for role in roles:
            if role in templates:
                print(f"{role} prompt")
                print(templates[role])
                print("----------------")
            else:
                print(" default prompt")
                print(templates["default"])
                print("----------------")

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx = sample["text"]
                true_idx = int(sample["label"])
                if not 0 <= true_idx < len(LABELS):
                    raise ValueError(
                        f"[Error] Task {task} has invalid label index {true_idx} "
                        f"(valid range: 0–{len(LABELS)-1}). Sample: {sample}"
                    )
        
                true_label = LABELS[true_idx]

                for role in roles:
                    prompt = construct_prompt(vc, templates, ctx, role, args.use_chat)
                    logits = vc.get_logits([prompt], return_hidden=False)
                    logits = logits[0, -1].cpu().numpy()

                    # Only in k options in the task
                    opt_logits = np.array([logits[i] for i in opt_ids])
                    probs = softmax_1d(opt_logits)
                    pred_idx = int(opt_logits.argmax())
                    pred_label = LABELS[pred_idx]
                    pred_prob = float(probs[pred_idx])

                    # sample
                    sample[rkey(role, "answer")] = pred_label
                    sample[rkey(role, "prob")] = pred_prob
                    sample[rkey(role, f"softmax_{task.replace(' ', '_')}")] = probs.tolist()
                    sample[rkey(role, f"logits_{task.replace(' ', '_')}")] = opt_logits.tolist()

                    # statistics
                    rs = role_stats[role]
                    rs["total"] += 1
                    if pred_label == true_label:
                        rs["correct"] += 1
                    elif pred_label == "E":     
                        rs["E_count"] += 1
                    else:
                        rs["invalid"] += 1

        # summary
        accuracy = {}
        for role, s in role_stats.items():
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0
            accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
            print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

        # save
        ans_file = ANS_DIR / f"{task.replace(' ', '_')}_{args.size}_answers.json"
        dump_json({"data": samples, "accuracy": accuracy}, ans_file)
        print("[Saved answers]", ans_file)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU-Pro role-based extraction with hidden-state saving")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., `8B`")
    parser.add_argument("--type", required=True, help="Role type identifier, affects prompt and output directories")
    parser.add_argument("--model_dir", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--ans_file", required=True, help="Subfolder name for outputs")
    parser.add_argument("--use_E", action="store_true", help="Use 5-choice template (A–E)")
    args = parser.parse_args()

    print("model: ", args.model)
    print("Loading model from:", args.model_dir)
    
    MMLU_PRO_DIR = Path("/data2/paveen/RolePlaying/components/mmlupro")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/{args.model}")
    HS_DIR  = Path(f"/data2/paveen/RolePlaying/components/hidden_states_{args.type}/{args.model}")
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    HS_DIR.mkdir(parents=True, exist_ok=True)

    main()