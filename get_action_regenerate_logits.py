#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply neuron-edit diffs and extract self-evaluated reasoning willingness (0–9)
for every role on every MMLU task.

"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
import utils

# ───────────────────── Helper ─────────────────────

def run_task(vc: VicundaModel, task: str, diff_mtx: np.ndarray, tail_len: int, role_type: str):
    """
    Run singnal task, apply diff_mtx, calculate 0-9 willingness.
    """
    # Template and ID
    templates = select_templates(suite="action")
    LABELS = templates["labels"]            # ["0","1",...,"9"]
    opt_ids = utils.option_token_ids(vc, LABELS)

    # Load data and construct roles
    data_path = os.path.join(MMLU_DIR, f"{task}.json")
    data = utils.load_json(data_path)
    roles = utils.make_characters(task, role_type)

    # Record template and statistics
    tmp_record = utils.record_template(roles, templates)
    stats = {r: {str(i): 0 for i in range(10)} | {"total": 0} for r in roles}

    for sample in tqdm(data, desc=task):
        ctx = sample.get("text", "")

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, use_chat=False)
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
            opt_logits = np.array([raw_logits[i] for i in opt_ids], dtype=np.float64)

            # softmax
            opt_logits -= opt_logits.max()     
            probs = np.exp(opt_logits)
            probs /= probs.sum()

            pred_idx = int(np.argmax(probs))
            pred_label = LABELS[pred_idx]      # "0".."9"
            pred_prob  = float(probs[pred_idx])

            rkey = role.replace(" ", "_")
            sample[f"score_{rkey}"]       = pred_label
            sample[f"score_prob_{rkey}"]  = pred_prob
            sample[f"score_dist_{rkey}"]  = probs.tolist()
            sample[f"logits_{rkey}"]      = opt_logits.tolist()

            # Update
            stats[role]["total"] += 1
            stats[role][pred_label] = stats[role].get(pred_label, 0) + 1

    # 6)Summary
    summary = {}
    for role, s in stats.items():
        total = s["total"]
        if total > 0:
            avg = sum(int(k) * v for k, v in s.items() if k.isdigit()) / total
        else:
            avg = 0.0
        summary[role] = {**s, "avg_score": round(avg, 3)}
        counts_only = {k: v for k, v in s.items() if k.isdigit()}
        print(f"{role:<25} avg_score={avg:5.2f}  counts={counts_only}")

    return data, summary, tmp_record


# ─────────────────────────── Main ───────────────────────────────

def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)

        diff_mtx = np.load(mask_path) * alpha   # 形如 (n_layers, hidden)
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))

        for task in TASKS:
            print(f"\n=== {task} | α={alpha} | layers={st}-{en} | TOP={TOP} ===")
            with torch.no_grad():
                updated_data, summary, tmp_record = run_task(
                    vc, task, diff_mtx, tail_len=args.tail_len, role_type=args.type
                )

            # Save
            out_dir = os.path.join(SAVE_ROOT, f"{args.model}_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "summary": summary, "template": tmp_record},
                          fw, ensure_ascii=False, indent=2)
            print("Saved →", out_path)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply neuron edits and extract action willingness (0–9).")

    parser.add_argument("--model",      type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir",  type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs",         type=str, default="qwen2.5")
    parser.add_argument("--size",       type=str, default="7B")
    parser.add_argument("--type",       type=str, default="non", help="role type (affects role names)")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs",    nargs="*", default=["4-16-22", "1-1-29"],help="List of alpha-start-end triplets, e.g. 4-16-22")
    parser.add_argument("--mask_type",  type=str, default="nmd", help="Mask type to load: nmd or random")
    parser.add_argument("--abs",        action="store_true")
    parser.add_argument("--ans_file",   type=str, default="answer_mdf_action")
    parser.add_argument("--tail_len",   type=int, default=1, help="How many last tokens to apply the diff to")
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])

    args = parser.parse_args()

    print("Model: ", args.model)
    print("Import model from:", args.model_dir)
    print("HS: ", args.hs)
    print("Mask Type:", args.mask_type)

    # Path
    MASK_DIR  = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    MMLU_DIR  = "/{args.data}/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/{args.data}/paveen/RolePlaying/components/{args.ans_file}"
    if args.abs:
        SAVE_ROOT += "_abs"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()