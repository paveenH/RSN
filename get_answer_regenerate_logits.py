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

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
import utils

# ───────────────────── Helper Functions ─────────────────────────


def run_task(
    vc: VicundaModel,
    task: str,
    diff_mtx: np.ndarray,
):
    """Run one task with a fixed diff_mtx, returning updated data + accuracy."""

    # template
    templates = select_templates(args.suite, args.use_E)  # choose template style
    LABELS = templates["labels"]
    opt_ids = utils.option_token_ids(vc, LABELS)

    # load data
    data_path = os.path.join(MMLU_DIR, f"{task}.json")
    data = utils.load_json(data_path)
    roles = utils.make_characters(task, args.type)

    tmp_record = utils.record_template(roles, templates)

    # stats accumulator
    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=task):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        true_lab = LABELS[true_idx]

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)

            # get raw logits after hooking in diff
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=args.tail_len)[0]
            # pick among options A–E
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(soft[pred_idx])

            role_key = role.replace(" ", "_")
            sample[f"answer_{role_key}"] = pred_lab
            sample[f"prob_{role_key}"] = pred_prb

            # Add whole prediction
            sample[f"softmax_{role_key}"] = [float(p) for p in soft]
            sample[f"logits_{role_key}"] = [float(l) for l in opt_logits]

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

    return data, accuracy, tmp_record


# ─────────────────────────── Main ───────────────────────────────


def main():

    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, f"{mask_name}")
        diff_mtx = np.load(mask_path) * alpha  # shape: (32, 4096)
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        for task in TASKS:
            print(f"\n=== {task} | α={alpha} | layers={st}-{en}| TOP={TOP} ===")

            with torch.no_grad():
                updated_data, accuracy, tmp_record = run_task(vc, task, diff_mtx)

            # save JSON
            out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "accuracy": accuracy, "template": tmp_record}, fw, ensure_ascii=False, indent=2)
            print("Saved →", out_path)

    print("\nAll tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vicunda model with neuron editing and logits output.")

    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22", "1-1-29"], help="List of alpha-start-end triplets, e.g. 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type to load: nmd or random")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--ans_file", type=str, default="answer_mdf")
    parser.add_argument("--E", dest="use_E", action="store_true")
    parser.add_argument("--use_chat", action="store_true", help="Use tokenizer.apply_chat_template for prompts")
    parser.add_argument("--tail_len", type=int, default=1, help="Number of last tokens to apply diff (default: 1)")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"], help="Prompt suite for MMLU-Pro")
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])

    args = parser.parse_args()

    print("Model: ", args.model)
    print("Import model from ", args.model_dir)
    print("HS: ", args.hs)
    print("Mask Type:", args.mask_type)

    # Path setup
    MASK_DIR = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    MMLU_DIR = f"/{args.data}/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/{args.data}/paveen/RolePlaying/components/{args.model}/{args.ans_file}"

    if args.abs:
        SAVE_ROOT += "_abs"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
