#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA (multiple_choice) → randomized-option JSON (MC1 & MC2)

- For each sample, synchronously shuffle choices & labels (and gold_indices).
- Rebuild the text with A/B/C/... letters mapped to the new order.
- Support multiple independent permutations in one run (num_permutations).
"""

from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import os, json, random, argparse
from copy import deepcopy

LETTER24 = [chr(ord("A") + i) for i in range(24)]

def _format_mc_text(question: str, options: List[str], letters: List[str]) -> str:
    text = (question or "").strip()
    K = min(len(options), len(letters))
    for i in range(K):
        text += f"\n{letters[i]}) {options[i]}"
    return text + "\n"

def _gold_indices_from_labels(labels: List[int]) -> List[int]:
    return [i for i, v in enumerate(labels) if int(v) == 1]

def _row_to_item(row: Dict[str, Any], target_key: str, max_choices: int = 10) -> Tuple[str, List[str], List[int], List[int], int]:
    """
    Parse one row into (text, choices, labels, gold_indices, num_options).
    Caps number of options to max_choices (default 10).
    """
    if target_key not in row or not isinstance(row[target_key], dict):
        raise ValueError(f"Row missing dict target '{target_key}'")
    tgt = row[target_key]
    choices: List[str] = list(tgt.get("choices", []))
    labels: List[int] = [int(x) for x in tgt.get("labels", [])]

    # Align & cut to same length
    if len(choices) != len(labels):
        m = min(len(choices), len(labels))
        choices = choices[:m]
        labels  = labels[:m]
    if len(choices) > max_choices:
        choices = choices[:max_choices]
        labels  = labels[:max_choices]

    gold_indices = _gold_indices_from_labels(labels)
    text = _format_mc_text(row.get("question", ""), choices, LETTER24)
    num_options = len(choices)
    return text, choices, labels, gold_indices, num_options

def _shuffle_once(item: Dict[str, Any], rnd: random.Random) -> Dict[str, Any]:
    """
    Given a base item {text, question, choices, labels, gold_indices, num_options},
    return a shuffled copy. We shuffle choices & labels with the same permutation,
    then recompute text & gold_indices. num_options remains len(choices).
    """
    new_item = deepcopy(item)

    choices = new_item["choices"]
    labels  = new_item["labels"]
    n = len(choices)

    # Construct a permutation
    perm = list(range(n))
    rnd.shuffle(perm)

    # Apply permutation
    choices_shuf = [choices[i] for i in perm]
    labels_shuf  = [labels[i]  for i in perm]

    # Recompute gold_indices from shuffled labels
    gold_indices_shuf = _gold_indices_from_labels(labels_shuf)

    # Rebuild text using new order and the stored raw question
    question = (new_item.get("question") or "").strip()
    text_shuf = _format_mc_text(question, choices_shuf, LETTER24)

    # Attach
    new_item["choices"] = choices_shuf
    new_item["labels"]  = labels_shuf
    new_item["gold_indices"] = gold_indices_shuf
    new_item["text"] = text_shuf
    new_item["perm"] = perm  # keep the permutation for traceability
    new_item["num_options"] = len(choices_shuf)  # keep consistent after shuffle
    return new_item

def export_truthfulqa_multiple_choice_shuffled(cache_dir: str, save_dir: str, split: str = "validation",
                                               num_permutations: int = 1, seed: int = 42):
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split=split, cache_dir=cache_dir)

    # Build base items (unshuffled)
    def make_base_item(row: Dict[str, Any], target_key: str, task_name: str) -> Dict[str, Any]:
        text, choices, labels, gold_indices, num_options = _row_to_item(row, target_key, max_choices=len(LETTER24))
        return {
            "task": task_name,
            "question": row.get("question", ""),   # keep the raw question to rebuild text robustly
            "text": text,                          # original rendered text
            "choices": choices,
            "labels": labels,
            "gold_indices": gold_indices,
            "num_options": num_options,            # <-- added
            # optional metadata you might want:
            "category": row.get("category", None),
            "id": row.get("id", None),
        }

    base_mc1 = [make_base_item(row, "mc1_targets", "TruthfulQA MC1") for row in ds]
    base_mc2 = [make_base_item(row, "mc2_targets", "TruthfulQA MC2") for row in ds]

    # For reproducibility across permutations
    for k in range(num_permutations):
        rnd = random.Random(seed + k)

        # MC1
        mc1_out = []
        for it in base_mc1:
            # Rebuild text from the stored 'question' to avoid cumulative parsing;
            # num_options is determined by the number of choices.
            it_clean = deepcopy(it)
            it_clean["text"] = _format_mc_text(it["question"], it["choices"], LETTER24)
            it_clean["num_options"] = len(it_clean["choices"])  # ensure consistency
            mc1_out.append(_shuffle_once(it_clean, rnd))

        out1 = os.path.join(save_dir, f"truthfulqa_mc1_{split}_shuf{k+1}.json" if num_permutations > 1
                            else f"truthfulqa_mc1_{split}_shuf.json")
        with open(out1, "w", encoding="utf-8") as f:
            json.dump(mc1_out, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved MC1 (perm {k+1}) → {out1}  (n={len(mc1_out)})")

        # MC2
        mc2_out = []
        for it in base_mc2:
            it_clean = deepcopy(it)
            it_clean["text"] = _format_mc_text(it["question"], it["choices"], LETTER24)
            it_clean["num_options"] = len(it_clean["choices"])  # ensure consistency
            mc2_out.append(_shuffle_once(it_clean, rnd))

        out2 = os.path.join(save_dir, f"truthfulqa_mc2_{split}_shuf{k+1}.json" if num_permutations > 1
                            else f"truthfulqa_mc2_{split}_shuf.json")
        with open(out2, "w", encoding="utf-8") as f:
            json.dump(mc2_out, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved MC2 (perm {k+1}) → {out2}  (n={len(mc2_out)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./.cache")
    parser.add_argument("--save_dir", type=str, default="./components/truthfulqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_permutations", type=int, default=1, help="How many independent shuffles to create")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    args = parser.parse_args()

    export_truthfulqa_multiple_choice_shuffled(
        cache_dir=args.cache_dir,
        save_dir=args.save_dir,
        split=args.split,
        num_permutations=args.num_permutations,
        seed=args.seed
    )