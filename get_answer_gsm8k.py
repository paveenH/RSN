#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K open-ended generation evaluation (baseline, no neuron editing).

Key difference from multiple-choice scripts:
- Uses model.generate() instead of model.get_logits()
- Extracts numeric answer from generated text via regex
- Evaluates by exact match of extracted number vs ground truth

@author: paveenhuang
"""

import re
import csv
import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_gsm8k
import utils


# ───────────────────── Answer extraction ─────────────────────

def extract_numeric_answer(text: str) -> str:
    """
    Extract the final numeric answer from generated text.
    Tries multiple patterns:
    1. #### <number>  (GSM8K standard format)
    2. The answer is <number>
    3. = <number> at the end
    4. Last number in the text
    """
    text = text.strip()

    # Pattern 1: #### <number>
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 2: "the answer is <number>"
    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 3: boxed{<number>}
    m = re.search(r"\\boxed\{([+-]?[\d,]+\.?\d*)\}", text)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 4: last number in text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def normalize_answer(ans: str) -> str:
    """Normalize a numeric answer for comparison."""
    ans = ans.strip().replace(",", "")
    try:
        # Convert to float then back to remove trailing zeros
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


def is_correct(pred: str, gold: str) -> bool:
    """Check if predicted answer matches ground truth."""
    return normalize_answer(pred) == normalize_answer(gold)


# ───────────────────── Main ─────────────────────

def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load GSM8K json
    all_samples: List[dict] = utils.load_json(DATA_DIR)
    print(f"Loaded {len(all_samples)} GSM8K samples.")

    # Templates
    templates = select_templates_gsm8k(suite=args.suite, cot=args.cot)

    # Roles
    custom_roles = None
    if args.roles:
        custom_roles = [r.strip() for r in args.roles.split(",")]
    roles = utils.make_characters("math", custom_roles)
    role_stats = {r: {"correct": 0, "total": 0} for r in roles}

    rows = []  # CSV rows

    with torch.no_grad():
        for sample in tqdm(all_samples, desc="GSM8K"):
            question = sample["question"]
            gold_answer = sample["answer"]

            for role in roles:
                # ─── Key difference: use construct_prompt with generation template ───
                prompt = utils.construct_prompt(vc, templates, question, role, args.use_chat)

                # ─── Key difference: generate() instead of get_logits() ───
                generated = vc.generate(
                    [prompt],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )[0]

                # ─── Key difference: extract number from text instead of argmax ───
                pred_answer = extract_numeric_answer(generated)
                correct = is_correct(pred_answer, gold_answer)

                # Attach to sample
                rk = role.replace(" ", "_")
                sample[f"generated_{rk}"] = generated
                sample[f"pred_answer_{rk}"] = pred_answer
                sample[f"correct_{rk}"] = correct

                # Stats
                rs = role_stats[role]
                rs["total"] += 1
                if correct:
                    rs["correct"] += 1

    # Summary
    print("\n=== GSM8K Results ===")
    for role, s in role_stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0.0
        print(f"{role:<25} acc={pct:5.2f}%  ({s['correct']}/{s['total']})")
        rows.append({
            "model": args.model,
            "size": args.size,
            "suite": args.suite,
            "cot": int(bool(args.cot)),
            "task": "gsm8k",
            "role": role,
            "correct": s["correct"],
            "total": s["total"],
            "accuracy_percentage": round(pct, 2),
        })

    # Save per-sample JSON
    task_dir = ANS_DIR / "orig"
    task_dir.mkdir(parents=True, exist_ok=True)
    ans_file = task_dir / f"gsm8k_{args.size}_answers.json"
    tmp_record = utils.record_template(roles, templates)
    utils.dump_json({"data": all_samples, "template": tmp_record}, ans_file)
    print(f"[Saved answers] {ans_file}")

    # Save CSV summary
    csv_file = ANS_DIR / f"summary_{args.model}_{args.size}.csv"
    fieldnames = [
        "model", "size", "suite", "cot",
        "task", "role", "correct", "total", "accuracy_percentage"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary CSV to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K generation evaluation (baseline)")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--size", "-s", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", required=True)
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"])
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--use_chat", action="store_true", help="Use tokenizer.apply_chat_template")
    parser.add_argument("--data", type=str, default="default", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--roles", type=str, default=None,
                        help="Comma-separated roles. E.g., 'neutral,math expert,non math expert'")
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate (GSM8K needs ~256-512 for CoT)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    print("model:", args.model)
    print("Loading model from:", args.model_dir)

    # Path setup
    if args.base_dir:
        BASE = Path(args.base_dir)
    else:
        BASE = Path(f"/{args.data}/paveen/RolePlaying/components")

    DATA_DIR = BASE / args.test_file
    ANS_DIR = BASE / args.model / args.ans_file
    ANS_DIR.mkdir(parents=True, exist_ok=True)

    main()
