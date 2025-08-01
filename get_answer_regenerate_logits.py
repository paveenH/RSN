#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel with neuron editing -> logits-based answer selection.
Loads the model once, applies diff matrices to hidden states, and for each prompt
directly reads out the last‐token logits to pick A/B/C/D/E.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse

import get_answer as ga
import get_answer_logits as gal
from llms import VicundaModel
from detection.task_list import TASKS

# ───────────────────── Helper Functions ─────────────────────────


def parse_configs(configs: list[str]):
    """
    Convert ['4-16-22', '1-1-29'] → [[4, (16, 22)], [1, (1, 29)]]
    """
    parsed = []
    for cfg in configs:
        try:
            alpha, start, end = map(int, cfg.strip().split("-"))
            parsed.append([alpha, (start, end)])
        except Exception:
            raise ValueError(f"Invalid config format: '{cfg}', should be alpha-start-end (e.g., 4-16-22)")
    return parsed


def run_task(
    vc: VicundaModel,
    template: str,
    task: str,
    diff_mtx: np.ndarray,
    opt_ids: list[int],
):
    """Run one task with a fixed diff_mtx, returning updated data + accuracy."""
    # load data
    data_path = os.path.join(MMLU_DIR, f"{task}.json")
    data = ga.load_json(data_path)
    roles = ga.make_characters(task, args.type)

    # stats accumulator
    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=task):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        if not (0 <= true_idx < len(LABELS)):
            continue
        true_lab = LABELS[true_idx]

        for role in roles:
            prompt = template.format(character=role, context=ctx)
            # get raw logits after hooking in diff
            raw_logits = vc.regenerate_logits([prompt], diff_mtx)[0]
            # pick among options A–E
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(soft[pred_idx])

            # write back answer
            key_ans = f"answer_{role.replace(' ', '_')}"
            key_prob = f"prob_{role.replace(' ', '_')}"
            sample[key_ans] = pred_lab
            sample[key_prob] = pred_prb

            # update stats
            st = stats[role]
            st["total"] += 1
            if pred_lab == true_lab:
                st["correct"] += 1
            elif pred_lab == "E":
                st["E_count"] += 1
            else:
                st["invalid"] += 1

    # accuracy summary
    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

    return data, accuracy


# ─────────────────────────── Main ───────────────────────────────


def main():
    
    ALPHAS_START_END_PAIRS = parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)
    
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    opt_ids = gal.option_token_ids(vc, LABELS)
    template = vc.template_mmlu_E

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, f"{mask_name}")
        diff_mtx = np.load(mask_path) * alpha # shape: (32, 4096)
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        for task in TASKS:
            print(f"\n=== {task} | α={alpha} | layers={st}-{en}| TOP={TOP} ===")
            with torch.no_grad():
                updated_data, accuracy = run_task(vc, template, task, diff_mtx, opt_ids)

            # save JSON
            out_dir = os.path.join(SAVE_ROOT, f"{args.model}_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "accuracy": accuracy}, fw, ensure_ascii=False, indent=2)
            print("Saved →", out_path)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    LABELS = ["A", "B", "C", "D", "E"]
    
    parser = argparse.ArgumentParser(description="Run Vicunda model with neuron editing and logits output.")

    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="+", default=["4-16-22", "1-1-29"], help="List of alpha-start-end triplets, e.g. 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type to load: nmd or random")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--ans_file", type=str, default="answer_mdf")
    
    args = parser.parse_args()
    
    print("Model: ", args.model)
    print("Import model from ", args.model_dir)
    print("HS: ", args.hs)
    print("Mask Type:", args.mask_type)

    # Path setup
    MASK_DIR = f"/data2/paveen/RolePlaying/components/mask/{args.model}_{args.type}"
    MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{args.ans_file}"
    
    if args.abs:
        SAVE_ROOT += "_abs"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
