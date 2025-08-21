#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 16:26:58 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FV per-layer sweep (random 5 tasks):
- Load a dense FV base from an NMD-100% file (shape: L-1, H; embedding layer removed).
- For EACH layer present in the file, build a diff_mtx with only that row nonzero (dense FV).
- Randomly sample 5 tasks from TASKS (reproducible via --seed).
- Use logits-based selection (A/B/C/D/E) and save JSON.

This script is independent from your previous runner.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
from numpy.random import default_rng

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
from utils import load_json, make_characters, option_token_ids, construct_prompt


def run_one_task(vc: VicundaModel,
                 task: str,
                 diff_mtx: np.ndarray,
                 use_E: bool,
                 use_chat: bool,
                 role_type: str,
                 tail_len: int,
                 mmlu_dir: str,
                 out_dir: str,
                 size_tag: str,
                 layer_idx_model: int):
    """
    Run one task with a fixed per-layer FV diff_mtx (only one layer non-zero).
    Save results to {out_dir}/{task}_{size}_answers_layer{layer}.json
    """
    templates = select_templates(use_E)
    opt_ids = option_token_ids(vc, templates["labels"])
    LABELS = templates["labels"]

    data_path = os.path.join(mmlu_dir, f"{task}.json")
    data = load_json(data_path)
    roles = make_characters(task, role_type)

    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=task):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        true_lab = LABELS[true_idx]

        for role in roles:
            prompt = construct_prompt(vc, templates, ctx, role, use_chat)

            # logits with FV injection (single-layer dense edit)
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()
            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(soft[pred_idx])

            role_key = role.replace(' ', '_')
            sample[f"answer_{role_key}"] = pred_lab
            sample[f"prob_{role_key}"] = pred_prb
            sample[f"softmax_{role_key}"] = [float(p) for p in soft]
            sample[f"logits_{role_key}"] = [float(l) for l in opt_logits]

            st = stats[role]
            st["total"] += 1
            if pred_lab == true_lab:
                st["correct"] += 1
            elif pred_lab == "E":
                st["E_count"] += 1
            else:
                st["invalid"] += 1

    # summarize
    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0.0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task}_{size_tag}_answers_layer{layer_idx_model}.json")
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump({"data": data, "accuracy": accuracy}, fw, ensure_ascii=False, indent=2)
    print("Saved →", out_path)


def main(args):
    # 1) Model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # 2) Load dense FV base (NMD-100%): shape (L-1, H); row i -> model layer (i+1)
    fv_full = np.load(args.fv_mask_path)  # (L-1, H)
    Lm1, H = fv_full.shape
    print(f"Loaded FV base: {args.fv_mask_path}  shape={fv_full.shape}  (rows=L-1, cols=H)")

    # 3) Randomly choose 5 tasks (reproducible)
    rng = default_rng(args.seed)
    num = min(5, len(TASKS))
    tasks = list(rng.choice(TASKS, size=num, replace=False))
    print(f"Randomly selected {num} tasks (seed={args.seed}): {tasks}")

    # 4) For EACH layer present in fv_full, build single-layer diff and run
    out_root = args.save_root
    os.makedirs(out_root, exist_ok=True)

    for row_idx in range(Lm1):
        layer_idx_model = row_idx + 1  # model layer index (embedding=0 removed)
        print(f"\n=== FV single-layer edit: layer={layer_idx_model} (row={row_idx}) | alpha={args.alpha} ===")

        # single-layer dense diff: only this row non-zero
        diff_mtx = np.zeros_like(fv_full, dtype=fv_full.dtype)
        diff_mtx[row_idx, :] = fv_full[row_idx, :] * args.alpha

        layer_out_dir = os.path.join(out_root, f"{args.model}_{args.alpha}_fv", f"layer_{layer_idx_model}")
        for task in tasks:
            with torch.no_grad():
                run_one_task(vc=vc,
                             task=task,
                             diff_mtx=diff_mtx,
                             use_E=args.use_E,
                             use_chat=args.use_chat,
                             role_type=args.type,
                             tail_len=args.tail_len,
                             mmlu_dir=args.mmlu_dir,
                             out_dir=layer_out_dir,
                             size_tag=args.size,
                             layer_idx_model=layer_idx_model)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FV per-layer sweep (random 5 tasks)")

    # model & prompt
    ap.add_argument("--model", type=str, default="llama3")
    ap.add_argument("--model_dir", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--size", type=str, default="8B")
    ap.add_argument("--type", type=str, default="non")  # 'non' or 'exp'
    ap.add_argument("--use_E", action="store_true")
    ap.add_argument("--use_chat", action="store_true")
    ap.add_argument("--tail_len", type=int, default=1)

    # FV base (写死路径默认就是你给的 NMD 100%)
    ap.add_argument("--fv_mask_path", type=str,
                    default="/data2/paveen/RolePlaying/components/mask/llama3_non_logits/nmd_100.0_1_33_8B.npy",
                    help="Dense FV base (L-1,H), embedding layer removed.")

    # alpha（强度），不再需要 start/end
    ap.add_argument("--alpha", type=float, default=4.0)

    # task 抽样与 IO
    ap.add_argument("--seed", type=int, default=2025, help="Random seed for task sampling")
    ap.add_argument("--mmlu_dir", type=str, default="/data2/paveen/RolePlaying/components/mmlu")
    ap.add_argument("--save_root", type=str, default="/data2/paveen/RolePlaying/components/answer_fv_random")

    args = ap.parse_args()
    main(args)