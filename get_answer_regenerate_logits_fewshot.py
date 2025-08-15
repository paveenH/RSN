#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel with neuron editing -> logits-based answer selection (few-shot version).
Loads the model once, applies diff matrices to hidden states, and for each prompt
directly reads out the last‐token logits to pick A/B/C/D.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
from utils import load_json, option_token_ids, parse_configs, build_fewshot_prefix, softmax_1d, dump_json


# ───────────────────── Helper Functions ─────────────────────────

def run_task(
    vc: VicundaModel,
    task: str,
    diff_mtx: np.ndarray,
):
    """Run one task with a fixed diff_mtx, returning updated data + accuracy."""
    data_path = os.path.join(MMLU_DIR, f"{task}.json")
    data = load_json(data_path)

    roles = ["vanilla"]
    stats = {r: {"correct": 0, "invalid": 0, "total": 0} for r in roles}
    
    templates = select_templates(False)
    LABELS = templates["labels"]
    template = templates["vanilla"]  # "{context}\nAnswer: "
    fewshot_prefix = build_fewshot_prefix(task=task, k=5)
    
    print(fewshot_prefix)
    print("------------------")
    print(template)
    
    
    opt_ids = option_token_ids(vc, LABELS)

    for sample in tqdm(data, desc=task):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        if not (0 <= true_idx < len(LABELS)):
            continue
        true_lab = LABELS[true_idx]

        for role in roles:
            # few-shot prompt
            question_block = template.format(context=ctx)
            prompt = f"{fewshot_prefix}\n{question_block}"

            # forward with neuron editing
            raw_logits = vc.regenerate_logits([prompt], diff_mtx)[0]

            # pick among options A–D
            opt_logits = np.array([raw_logits[i] for i in opt_ids])
            probs = softmax_1d(opt_logits)

            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(probs[pred_idx])

            # record
            role_key = role.replace(' ', '_')
            sample[f"answer_{role_key}"] = pred_lab
            sample[f"prob_{role_key}"] = pred_prb
            sample[f"softmax_{role_key}"] = probs.tolist()
            sample[f"logits_{role_key}"] = opt_logits.tolist()

            # update stats
            st = stats[role]
            st["total"] += 1
            if pred_lab == true_lab:
                st["correct"] += 1
            else:
                st["invalid"] += 1

    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']})")

    return data, accuracy


def main():
    ALPHAS_START_END_PAIRS = parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, f"{mask_name}")
        diff_mtx = np.load(mask_path) * alpha
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))

        for task in TASKS:
            print(f"\n=== {task} | α={alpha} | layers={st}-{en}| TOP={TOP} ===")

            with torch.no_grad():
                updated_data, accuracy = run_task(vc, task, diff_mtx)

            out_dir = os.path.join(SAVE_ROOT, f"{args.model}_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
            dump_json({"data": updated_data, "accuracy": accuracy}, Path(out_path))
            print("Saved →", out_path)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vicunda model with neuron editing and few-shot logits output.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--ans_file", type=str, required=True)
    args = parser.parse_args()

    MASK_DIR = f"/data2/paveen/RolePlaying/components/mask/{args.model}_{args.type}_logits"
    MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{args.ans_file}"
    if args.abs:
        SAVE_ROOT += "_abs"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()