#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriviaQA open-ended generation evaluation with neuron editing (regenerate).

Key difference from get_answer_triviaqa.py:
- Uses model.regenerate() instead of model.generate()
- Loads NMD diff masks and applies them during generation

@author: paveenhuang
"""

import os
import re
import gc
import csv
import json
import string
import argparse
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_triviaqa
import utils


# ───────────────────── Answer extraction & matching ─────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation & articles."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_short_answer(text: str) -> str:
    """
    Extract the short answer from generated text.
    """
    text = text.strip()
    if not text:
        return ""

    first_line = text.split("\n")[0].strip()

    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*(.+)", first_line)
    if m:
        return m.group(1).strip().rstrip(".")

    m = re.match(r"^([^.]+)\.", first_line)
    if m and len(m.group(1)) < 200:
        return m.group(1).strip()

    return first_line


def is_correct(pred: str, gold: str, aliases: List[str]) -> bool:
    """Check if prediction matches any acceptable answer."""
    pred_norm = normalize_text(pred)
    if pred_norm == normalize_text(gold):
        return True
    for alias in aliases:
        if pred_norm == normalize_text(alias):
            return True
    if normalize_text(gold) in pred_norm:
        return True
    for alias in aliases:
        if normalize_text(alias) in pred_norm:
            return True
    return False


# ───────────────────── Per-config runner ─────────────────────

def run_triviaqa_regenerate(
    vc: VicundaModel,
    samples: List[dict],
    diff_mtx: np.ndarray,
    templates: dict,
    roles: List[str],
):
    """
    Run TriviaQA with neuron editing applied during generation.
    """
    stats = {r: {"correct": 0, "total": 0} for r in roles}

    for sample in tqdm(samples, desc="TriviaQA-regen"):
        question = sample["question"]
        gold_answer = sample["answer"]
        aliases = sample.get("aliases", [])

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, question, role, args.use_chat)

            generated = vc.regenerate(
                [prompt],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=diff_mtx,
            )[0]

            pred_answer = extract_short_answer(generated)
            correct = is_correct(pred_answer, gold_answer, aliases)

            rk = role.replace(" ", "_")
            sample[f"generated_{rk}"] = generated
            sample[f"pred_answer_{rk}"] = pred_answer
            sample[f"correct_{rk}"] = correct

            st = stats[role]
            st["total"] += 1
            if correct:
                st["correct"] += 1

    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  ({s['correct']}/{s['total']})")

    return samples, accuracy


# ───────────────────── Main ─────────────────────

def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load TriviaQA data
    all_samples = utils.load_json(DATA_DIR)
    print(f"Loaded {len(all_samples)} TriviaQA samples.")

    # Templates
    templates = select_templates_triviaqa(suite=args.suite, cot=args.cot)

    # Roles
    custom_roles = None
    if args.roles:
        custom_roles = [r.strip() for r in args.roles.split(",")]
    roles = utils.make_characters("trivia", custom_roles)

    # Outer loop: each alpha / layer range
    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        diff_mtx = np.load(mask_path) * alpha
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        print(f"\n=== alpha={alpha} | layers={st}-{en} | TOP={TOP} ===")

        import copy
        config_samples = copy.deepcopy(all_samples)

        with torch.no_grad():
            updated_data, accuracy = run_triviaqa_regenerate(
                vc=vc,
                samples=config_samples,
                diff_mtx=diff_mtx,
                templates=templates,
                roles=roles,
            )

        # Save JSON
        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"triviaqa_{args.size}_answers_{TOP}_{st}_{en}.json")
        tmp_record = utils.record_template(roles, templates)
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(
                {"data": updated_data, "accuracy": accuracy, "template": tmp_record},
                fw, ensure_ascii=False, indent=2,
            )
        print("Saved ->", out_path)

        # Save CSV
        csv_rows = []
        for role, s in accuracy.items():
            csv_rows.append({
                "model": args.model,
                "size": args.size,
                "alpha": alpha,
                "start": st,
                "end": en,
                "TOP": TOP,
                "task": "triviaqa",
                "role": role,
                "correct": s["correct"],
                "total": s["total"],
                "accuracy_percentage": s["accuracy_percentage"],
                "suite": args.suite,
                "cot": int(bool(args.cot)),
            })

        csv_path = os.path.join(out_dir, f"summary_triviaqa_{args.model}_{args.size}_{TOP}_{st}_{en}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model", "size", "alpha", "start", "end", "TOP",
                    "task", "role", "correct", "total", "accuracy_percentage",
                    "suite", "cot",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved CSV] {csv_path}")

        del csv_rows, config_samples
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriviaQA generation with neuron editing (regenerate)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"],
                        help="List of alpha-start-end triplets")
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", type=str, default="answer_mdf_triviaqa")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"])
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--data", type=str, default="default", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--roles", type=str, default=None)
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    print("Model:", args.model)
    print("Import model from:", args.model_dir)
    print("HS:", args.hs)
    print("Mask Type:", args.mask_type)

    # Path setup
    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    DATA_DIR = os.path.join(BASE, args.test_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
