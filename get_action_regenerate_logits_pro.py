#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run action-style scoring (0–9) with neuron editing (mdf) on MMLU-Pro JSON.
- Uses regenerate_logits with diff_mtx
- Collects per-role distributions and summary stats (mean/std/entropy/tails)
"""

import os
import json
import csv
import numpy as np
import torch
from tqdm import tqdm
import argparse
import math

from llms import VicundaModel
from template import select_templates_pro
import utils


def run_task_action_mdf(vc, task, samples, diff_mtx):
    """Run one task with neuron editing → action logits (0–9)."""
    templates = select_templates_pro(suite="action")
    LABELS = templates["labels"]              # ["0"... "9"]
    opt_ids = utils.option_token_ids(vc, LABELS)
    roles = utils.make_characters(task.replace(" ", "_"), args.type)
    tmp_record = utils.record_template(roles, templates)

    # stats accumulator
    role_stats = {r: {str(i): 0 for i in range(10)} | {"total": 0} for r in roles}

    for sample in tqdm(samples, desc=task):
        ctx = sample["text"]

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, False)

            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=args.tail_len)[0]
            opt_logits = np.array([raw_logits[i] for i in opt_ids], dtype=float)

            # softmax
            opt_logits -= opt_logits.max()
            probs = np.exp(opt_logits)
            probs /= probs.sum()

            pred_idx = int(np.argmax(opt_logits))
            pred_label = LABELS[pred_idx]
            pred_prob = float(probs[pred_idx])

            rk = role.replace(" ", "_")
            sample[f"score_{rk}"] = pred_label
            sample[f"score_prob_{rk}"] = pred_prob
            sample[f"score_dist_{rk}"] = probs.tolist()
            sample[f"logits_{rk}"] = opt_logits.tolist()

            rs = role_stats[role]
            rs["total"] += 1
            rs[pred_label] = rs.get(pred_label, 0) + 1

    return samples, role_stats, tmp_record


def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    all_samples = utils.load_json(DATA_DIR)
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} tasks in MMLU-Pro JSON.")

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        diff_mtx = np.load(mask_path) * alpha
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")

        csv_rows = []

        for task in tasks:
            task_samples = [s for s in all_samples if s["task"] == task]
            if not task_samples:
                continue

            print(f"\n--- Task: {task} ---")
            with torch.no_grad():
                updated_data, role_stats, tmp_record = run_task_action_mdf(vc, task, task_samples, diff_mtx)

            # summary
            summary = {}
            for role, s in role_stats.items():
                total = int(s["total"])
                counts = [int(s[str(i)]) for i in range(10)]
                if total > 0:
                    mean = sum(i * c for i, c in enumerate(counts)) / total
                    var = sum(((i - mean) ** 2) * c for i, c in enumerate(counts)) / total
                    std = math.sqrt(var)
                    dist = np.array(counts, dtype=float) / total
                    ent = utils.entropy_bits(dist)
                    top_score = int(np.argmax(counts))
                    top_ratio = max(counts) / total
                    tail_low = sum(counts[0:3]) / total
                    tail_high = sum(counts[8:10]) / total
                else:
                    mean = std = ent = top_ratio = tail_low = tail_high = 0.0
                    top_score = 0

                summary[role] = {**{str(i): counts[i] for i in range(10)},
                                 "total": total, "avg_score": round(mean, 3)}

                print(f"{role:<25} avg_score={mean:5.2f} counts={counts}")

                row = {
                    "model": args.model,
                    "size": args.size,
                    "alpha": alpha,
                    "start": st,
                    "end": en,
                    "TOP": TOP,
                    "task": task,
                    "role": role,
                    "total": total,
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "entropy_bits": round(ent, 6),
                    "top_score": top_score,
                    "top_ratio": round(top_ratio, 6),
                    "tail_low_0_2": round(tail_low, 6),
                    "tail_high_8_9": round(tail_high, 6),
                }
                for i in range(10):
                    row[f"counts_{i}"] = counts[i]
                csv_rows.append(row)

            # Save JSON
            out_dir = os.path.join(SAVE_ROOT, f"{args.model}_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task.replace(' ', '_')}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "summary": summary, "template": tmp_record},
                          fw, ensure_ascii=False, indent=2)
            print("Saved →", out_path)

        # Save CSV
        csv_path = os.path.join(SAVE_ROOT, f"{args.model}_{alpha}",
                                f"summary_{args.model}_{args.size}_{TOP}_{st}_{en}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["model","size","alpha","start","end","TOP","task","role",
                          "total","mean","std","entropy_bits","top_score","top_ratio",
                          "tail_low_0_2","tail_high_8_9"] + [f"counts_{i}" for i in range(10)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved CSV] {csv_path}")

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vicunda model (MMLU-Pro) with neuron editing → action logits")
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"], help="alpha-start-end triplets")
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", type=str, default="answer_mdf_action")
    parser.add_argument("--tail_len", type=int, default=1)
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    args = parser.parse_args()

    print("Model:", args.model)
    print("Import model from", args.model_dir)
    print("HS:", args.hs)

    DATA_DIR = f"/{args.data}/paveen/RolePlaying/components/{args.test_file}"
    MASK_DIR = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    SAVE_ROOT = f"/{args.data}/paveen/RolePlaying/components/{args.ans_file}"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()