#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FV per-layer sweep (random 5 tasks).
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
from numpy.random import default_rng

from llms import VicundaModel
from template import select_templates
from utils import load_json, make_characters, option_token_ids, construct_prompt

TASKS = ["college_computer_science", "us_foreign_policy", "management", "jurisprudence"]

def run_one_task(
    vc,
    task,
    diff_mtx,
    use_E,
    role_type,
    out_dir,
    layer_idx_model,
    *,
    tail_len: int = 1,
    use_chat: bool = False,
):
    """
    Run one task with a fixed per-layer FV diff_mtx (only one layer non-zero).
    Save results to {out_dir}/{task}_{args.size}_answers_{TOP}_{layer}_{layer+1}.json
    """
    TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))

    templates = select_templates(use_E)
    LABELS = templates["labels"]
    opt_ids = option_token_ids(vc, LABELS)

    data_path = os.path.join(MMLU_DIR, f"{task}.json")
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

            role_key = role.replace(" ", "_")
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
        print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{task}_{args.size}_answers_{TOP}_{layer_idx_model}_{layer_idx_model+1}.json"
    )
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump({"data": data, "accuracy": accuracy}, fw, ensure_ascii=False, indent=2)
    print("Saved →", out_path)


def main(args):
    # Model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # FV base (dense): (L-1, H)
    fv_full = np.load(MASK_DIR)
    Lm1, H = fv_full.shape
    print(f"Loaded FV base: {MASK_DIR}  shape={fv_full.shape}")

    # Random 5 tasks
    print(f"Randomly selected 5 tasks : {TASKS}")

    # Per-layer sweep
    os.makedirs(SAVE_ROOT, exist_ok=True)

    for row_idx in range(Lm1):
        layer_idx_model = row_idx + 1  # skip embedding (layer 0)
        print(f"\n=== FV single-layer edit: layer={layer_idx_model} | alpha={ALPHA} ===")

        diff_mtx = np.zeros_like(fv_full, dtype=fv_full.dtype)
        diff_mtx[row_idx, :] = fv_full[row_idx, :] * ALPHA

        layer_out_dir = os.path.join(SAVE_ROOT, f"{args.model}_{ALPHA}")
        for task in TASKS:
            with torch.no_grad():
                run_one_task(
                    vc=vc,
                    task=task,
                    diff_mtx=diff_mtx,
                    use_E=args.use_E,
                    role_type=args.type,
                    out_dir=layer_out_dir,
                    layer_idx_model=layer_idx_model,
                    tail_len=1,        
                    use_chat=False  
                )

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FV per-layer sweep (random 5 tasks)")

    # Basic
    ap.add_argument("--model", type=str, default="llama3")
    ap.add_argument("--model_dir", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--hs", type=str, default="llama3")
    ap.add_argument("--size", type=str, default="8B")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--type", type=str, default="non")
    ap.add_argument("--use_E", action="store_true")
    ap.add_argument("--ans_file", type=str, default="answer_mdf")
    ap.add_argument("--mask_name", type=str, default="nmd_100.0_1_33_8B.npy")

    args = ap.parse_args()

    ALPHA = 1
    MASK_DIR = f"/data2/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits/{args.mask_name}"
    MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{args.ans_file}"

    print("Model:", args.model)
    print("Import model from:", args.model_dir)
    print("HS:", args.hs)

    main(args)