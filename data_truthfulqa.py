#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA (multiple_choice) → unified JSON (MC1 & MC2) with ALL labels.

"""

from typing import List, Dict, Any, Tuple
from datasets import load_dataset
import os, json

LETTER10 = ["A","B","C","D","E","F","G","H","I","J"]

def _format_mc_text(question: str, options: List[str], letters: List[str]) -> str:
    text = question.strip()
    K = min(len(options), len(letters))
    for i in range(K):
        text += f"\n{letters[i]}) {options[i]}"
    return text + "\n"

def _gold_indices_from_labels(labels: List[int]) -> List[int]:
    return [i for i, v in enumerate(labels) if int(v) == 1]

def _first_gold_or_zero(gold_indices: List[int]) -> int:
    return gold_indices[0] if gold_indices else 0

def _row_to_item(row: Dict[str, Any], target_key: str, max_choices: int = 10) -> Tuple[str, List[str], List[int], List[int]]:

    if target_key not in row or not isinstance(row[target_key], dict):
        raise ValueError(f"Row missing dict target '{target_key}'")

    tgt = row[target_key]
    choices: List[str] = list(tgt.get("choices", []))
    labels: List[int] = [int(x) for x in tgt.get("labels", [])]

    if len(choices) != len(labels):
        m = min(len(choices), len(labels))
        choices = choices[:m]
        labels = labels[:m]

    if len(choices) > max_choices:
        choices = choices[:max_choices]
        labels  = labels[:max_choices]

    gold_indices = _gold_indices_from_labels(labels)
    text = _format_mc_text(row.get("question", ""), choices, LETTER10)
    return text, choices, labels, gold_indices

def export_truthfulqa_multiple_choice(cache_dir: str, save_dir: str, split: str = "validation"):
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split=split, cache_dir=cache_dir, trust_remote_code=True)

    def make_item(row: Dict[str, Any], target_key: str, task_name: str) -> Dict[str, Any]:
        text, choices, labels, gold_indices = _row_to_item(row, target_key, max_choices=len(LETTER10))
        return {
            "task": task_name,
            "text": text,                             # question + answers
            "choices": choices,                       # 
            "labels": labels,                         # 
            "gold_indices": gold_indices,             # 
        }

    # MC1
    mc1_out: List[Dict[str, Any]] = [make_item(row, "mc1_targets", "TruthfulQA MC1") for row in ds]
    out1 = os.path.join(save_dir, f"truthfulqa_mc1_{split}.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(mc1_out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved TruthfulQA MC1 → {out1}  (n={len(mc1_out)})")

    # MC2
    mc2_out: List[Dict[str, Any]] = [make_item(row, "mc2_targets", "TruthfulQA MC2") for row in ds]
    out2 = os.path.join(save_dir, f"truthfulqa_mc2_{split}.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(mc2_out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved TruthfulQA MC2 → {out2}  (n={len(mc2_out)})")

if __name__ == "__main__":
    cache_dir = "./.cache"
    save_dir  = "./components/truthfulqa"
    split = "validation"  
    export_truthfulqa_multiple_choice(cache_dir, save_dir, split)