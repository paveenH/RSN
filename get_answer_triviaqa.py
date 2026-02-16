#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriviaQA open-ended generation evaluation (baseline, no neuron editing).

Key difference from GSM8K:
- Answer is free-form text (not a number)
- Evaluation uses normalized exact match + alias matching
- Extracts the first sentence/short answer from generated text

@author: paveenhuang
"""

import re
import csv
import string
import argparse
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_triviaqa
import utils


# ───────────────────── Answer extraction & matching ─────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation & articles."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_short_answer(text: str) -> str:
    """
    Extract the short answer from generated text.
    Takes the first line or first sentence as the answer.
    """
    text = text.strip()
    if not text:
        return ""

    # Take first line
    first_line = text.split("\n")[0].strip()

    # If first line contains "the answer is ...", extract it
    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*(.+)", first_line)
    if m:
        return m.group(1).strip().rstrip(".")

    # Take first sentence (up to first period, if short enough)
    m = re.match(r"^([^.]+)\.", first_line)
    if m and len(m.group(1)) < 200:
        return m.group(1).strip()

    return first_line


def is_correct(pred: str, gold: str, aliases: List[str]) -> bool:
    """
    Check if prediction matches any of the acceptable answers.
    Uses normalized exact match.
    """
    pred_norm = normalize_text(pred)
    # Check canonical answer
    if pred_norm == normalize_text(gold):
        return True
    # Check aliases
    for alias in aliases:
        if pred_norm == normalize_text(alias):
            return True
    # Also check if gold/alias is contained in prediction (for longer answers)
    if normalize_text(gold) in pred_norm:
        return True
    for alias in aliases:
        if normalize_text(alias) in pred_norm:
            return True
    return False


# ───────────────────── Main ─────────────────────

def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load TriviaQA json
    all_samples: List[dict] = utils.load_json(DATA_DIR)
    print(f"Loaded {len(all_samples)} TriviaQA samples.")

    # Templates
    templates = select_templates_triviaqa(suite=args.suite, cot=args.cot)

    # Roles
    custom_roles = None
    if args.roles:
        custom_roles = [r.strip() for r in args.roles.split(",")]
    roles = utils.make_characters("trivia", custom_roles)
    role_stats = {r: {"correct": 0, "total": 0} for r in roles}

    rows = []  # CSV rows

    with torch.no_grad():
        for sample in tqdm(all_samples, desc="TriviaQA"):
            question = sample["question"]
            gold_answer = sample["answer"]
            aliases = sample.get("aliases", [])

            for role in roles:
                prompt = utils.construct_prompt(vc, templates, question, role, args.use_chat)

                generated = vc.generate(
                    [prompt],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )[0]

                pred_answer = extract_short_answer(generated)
                correct = is_correct(pred_answer, gold_answer, aliases)

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
    print("\n=== TriviaQA Results ===")
    for role, s in role_stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0.0
        print(f"{role:<25} acc={pct:5.2f}%  ({s['correct']}/{s['total']})")
        rows.append({
            "model": args.model,
            "size": args.size,
            "suite": args.suite,
            "cot": int(bool(args.cot)),
            "task": "triviaqa",
            "role": role,
            "correct": s["correct"],
            "total": s["total"],
            "accuracy_percentage": round(pct, 2),
        })

    # Save per-sample JSON
    task_dir = ANS_DIR / "orig"
    task_dir.mkdir(parents=True, exist_ok=True)
    ans_file = task_dir / f"triviaqa_{args.size}_answers.json"
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
    parser = argparse.ArgumentParser(description="TriviaQA generation evaluation (baseline)")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--size", "-s", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", required=True)
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"])
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--data", type=str, default="default", choices=["data1", "data2"])
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--roles", type=str, default=None,
                        help="Comma-separated roles. E.g., 'neutral,trivia expert'")
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate (TriviaQA answers are short)")
    parser.add_argument("--temperature", type=float, default=0.0)
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
