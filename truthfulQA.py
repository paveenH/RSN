#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA (multiple_choice) → unified JSON (MC1 & MC2) for your pipeline.

- Builds text as: question + enumerated options ("A) ...", ...).
- Returns single-label "label" (int). For MC1 one-hot；MC2 取第一个正标签。
- Saves: truthfulqa_mc1_<split>.json, truthfulqa_mc2_<split>.json
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

def _gold_from_labels(labels: List[int]) -> int:
    """Return first index with label==1; fallback to 0."""
    for i, v in enumerate(labels):
        if int(v) == 1:
            return i
    return 0

def _row_to_item(row: Dict[str, Any], target_key: str, max_choices: int = 10) -> Tuple[str, int]:
    """
    Convert one row to (text, gold_idx) using target_key in {"mc1_targets","mc2_targets"}.
    Cuts options to max_choices and re-computes gold_idx in the truncated list.
    """
    tgt = row[target_key]  # dict with "choices" and "labels"
    choices: List[str] = list(tgt["choices"])
    labels: List[int] = list(tgt["labels"])

    # truncate to A..J
    if len(choices) > max_choices:
        choices = choices[:max_choices]
        labels  = labels[:max_choices]

    gold_idx = _gold_from_labels(labels)
    text = _format_mc_text(row["question"], choices, LETTER10)
    return text, gold_idx

def export_truthfulqa_multiple_choice(cache_dir: str, save_dir: str, split: str = "validation"):
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset("truthful_qa", "multiple_choice", split=split, cache_dir=cache_dir, trust_remote_code=True)

    # MC1
    mc1_out: List[Dict[str, Any]] = []
    for row in ds:
        text, gold = _row_to_item(row, "mc1_targets", max_choices=len(LETTER10))
        mc1_out.append({
            "task": "TruthfulQA MC1",
            "category": row.get("category", ""),   # may be empty in multiple_choice config
            "text": text,
            "label": int(gold),
        })
    out1 = os.path.join(save_dir, f"truthfulqa_mc1_{split}.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(mc1_out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved TruthfulQA MC1 → {out1}  (n={len(mc1_out)})")

    # MC2
    mc2_out: List[Dict[str, Any]] = []
    for row in ds:
        text, gold = _row_to_item(row, "mc2_targets", max_choices=len(LETTER10))
        mc2_out.append({
            "task": "TruthfulQA MC2",
            "category": row.get("category", ""),
            "text": text,
            "label": int(gold),
        })
    out2 = os.path.join(save_dir, f"truthfulqa_mc2_{split}.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(mc2_out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved TruthfulQA MC2 → {out2}  (n={len(mc2_out)})")

if __name__ == "__main__":
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/truthfulqa"
    split = "validation"  # TruthfulQA only has validation, no test.

    export_truthfulqa_multiple_choice(cache_dir, save_dir, split)