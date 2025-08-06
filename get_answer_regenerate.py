#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel with neuron editing across multiple tasks.
Loads the model once and processes all combinations of tasks, α values, and layer ranges 
by injecting diff matrices into generation. Selects highest‐prob answer via generation.
@author: paveenhuang
"""

import os
import json
import numpy as np
from tqdm import tqdm
import torch
import argparse
from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
from utils import cleaning, load_json, make_characters, extract_full_correct_text, update, parse_configs

# ───────────────────── Helper functions ─────────────────────────


def generate_answer_diff(vc, prompt: str, diff_mtx: np.ndarray, short: int) -> str:
    """Generate with neuron‐diff injected, return cleaned single‐token answer."""
    out = vc.regenerate([prompt], diff_matrices=diff_mtx, max_new_tokens=short)[0]
    return cleaning(out)


def handle_invalid_answer_diff(
    vc, prompt: str, true_text: str, true_label: str, diff_mtx: np.ndarray, long: int
) -> tuple[str, bool, bool]:
    """Second‐pass rescue generation with longer max tokens."""
    out = (
        vc.regenerate([prompt], diff_matrices=diff_mtx, max_new_tokens=long)[0]
        .replace("<|assistant|>", "")
        .replace("\u200b", "")
        .strip()
        .upper()
    )
    extracted = cleaning(out)

    if extracted in LABELS:
        if extracted == true_label:
            return ("[Add]" + extracted + out, True, False)
        return (extracted + out, False, False)

    if extracted == "E":
        return ("[Add]" + out, False, True)
    if true_text and true_text.lower() in out.lower():
        return ("[Add]" + out + "---" + true_text, True, False)
    if "i am not sure" in out.lower():
        return ("[Add]" + out, False, True)
    
    return (out, False, False) 


def run_task(vc: VicundaModel, templates: dict, task: str, diff_mtx: np.ndarray, SHORT: int, LONG: int):
    """
    For a single MMLU task and a fixed diff_mtx, run generation for each role,
    do rescue if needed, and accumulate accuracy/E/invalid stats.
    """
    LABELS = templates["labels"]
    default_t = templates["default"]
    neutral_t = templates["neutral"]
    neg_t = templates["neg"]
    vanilla_t = templates["vanilla"]

    data = load_json(os.path.join(MMLU_DIR, f"{task}.json"))
    chars = make_characters(task, TYPE)
    acc = {c: {"correct": 0, "E": 0, "invalid": 0, "total": 0} for c in chars}

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data, desc=task)):
            ctx = sample["text"]
            true_idx = sample["label"]
            if not 0 <= true_idx < len(LABELS):
                continue
            true_label = LABELS[true_idx]
            true_text = extract_full_correct_text(ctx, true_idx, LABELS)

            for ch in chars:
                # choose prompt template
                if ch == "norole":
                    prompt = neutral_t.format(context=ctx)
                elif ch == "vanilla":
                    prompt = vanilla_t.format(context=ctx)
                elif "not" in ch:
                    prompt = neg_t.format(character=ch, context=ctx)
                else:
                    prompt = default_t.format(character=ch, context=ctx)

                # first‐pass generation
                ans = generate_answer_diff(vc, prompt, diff_mtx, SHORT)

                # rescue invalid
                if ans not in LABELS and ans != "E":
                    ans, is_corr, is_E = handle_invalid_answer_diff(vc, prompt, true_text, true_label, diff_mtx, LONG)
                    if is_corr:
                        status = "correct"
                        tqdm.write(f"[{idx}][{ch}] '{ans}' → Correct")
                    elif is_E:
                        status = "E"
                        tqdm.write(f"[{idx}][{ch}] '{ans}' → E")
                    else:
                        status = "invalid"
                        tqdm.write(f"[{idx}][{ch}] '{ans}' → Invalid")
                else:
                    status = "correct" if ans == true_label else ("E" if ans == "E" else "invalid")

                # accumulate stats
                acc[ch]["total"] += 1
                update(acc, ch, status)
                sample[f"answer_{ch.replace(' ', '_')}"] = ans

    # build summary
    summary = {}
    for ch, stats in acc.items():
        total = stats["total"]
        correct = stats["correct"]
        pct = (correct / total * 100) if total else 0.0
        summary[ch] = {
            "correct": correct,
            "E_count": stats["E"],
            "invalid": stats["invalid"],
            "total": total,
            "accuracy_percentage": round(pct, 2),
        }
    return data, summary


# ─────────────────────────── Main ───────────────────────────────


def main():
    # 1) parse configs
    pairs = parse_configs(args.configs)  # list of [alpha, (start,end)]
    templates = select_templates(args.use_E)

    # 2) load model
    print(f"Loading model from {args.model_dir} …")
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # 3) iterate all α and layer ranges
    for alpha, (st, en) in pairs:
        mask_suffix = "_abs" if args.abs else ""
        mask_file = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_file)
        diff_mtx = np.load(mask_path) * alpha

        for task in TASKS:
            print(f"\n=== Task={task} | α={alpha} | layers={st}-{en} | TOP mask applied ===")
            # run and save
            data, acc = run_task(vc, templates, task, diff_mtx, args.short, args.long)

            out_dir = os.path.join(SAVE_ROOT, f"{args.model}_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"data": data, "accuracy": acc}, f, ensure_ascii=False, indent=2)

            # print summary
            for ch, r in acc.items():
                print(
                    f"{ch:>18}: {r['accuracy_percentage']}% "
                    f"(correct {r['correct']}/{r['total']}, E {r['E_count']}, invalid {r['invalid']})"
                )

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VicundaModel generation‐based neuron editing across MMLU tasks")
    parser.add_argument("--model", "-m", required=True, help="Model name for outputs")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., 4B")
    parser.add_argument("--type", required=True, help="Role type identifier")
    parser.add_argument("--model_dir", required=True, help="Model checkpoint directory")
    parser.add_argument("--ans_file", required=True, help="Base output folder name")
    parser.add_argument("--configs", nargs="+", required=True, help="List of alpha-start-end triplets, e.g. 4-7-15")
    parser.add_argument("--mask_type", required=True, help="Mask type: nmd, diff_random, etc.")
    parser.add_argument("--percentage", type=float, default=0.5, help="Fraction for TOP-k in mask name (e.g. 0.5)")
    parser.add_argument("--abs", action="store_true", help="Use _abs mask suffix")
    parser.add_argument("--use_E", action="store_true", help="Use five-choice template (A–E)")
    parser.add_argument("--short", type=int, default=2, help="Max tokens for first-pass generation")
    parser.add_argument("--long", type=int, default=12, help="Max tokens for rescue generation")

    args = parser.parse_args()

    # constants setup
    MODEL = args.model
    SIZE = args.size
    TYPE = args.type
    MASK_DIR = f"/data2/paveen/RolePlaying/components/mask/{MODEL}_{TYPE}"
    MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{args.ans_file}"

    # re-export for helper scope
    LABELS = select_templates(args.use_E)["labels"]

    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
