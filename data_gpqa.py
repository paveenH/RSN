#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 11:58:02 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPQA → MMLU-Pro-like JSON (no shuffle)

- Supports both columnar schema (Question, Correct/Incorrect Answer X) and
  key-based schema (question/options/answer) if encountered.
- No shuffling; preserves original option order; uses given gold index.
- Works with configs: gpqa_main, gpqa_diamond, gpqa_extended, gpqa_experts
  (all of them commonly provide split='train' for evaluation use).

Usage:
  python gpqa_to_mmlupro.py --config gpqa_main --split train \
      --out /path/to/gpqa_main_train.json --limit 0
"""

import os
import json
import argparse
from typing import List, Dict, Any
from datasets import load_dataset

LETTER = [chr(ord("A")+i) for i in range(26)]

def build_text(question: str, options: List[str]) -> str:
    question = (question or "").strip()
    lines = [question]
    for i, opt in enumerate(options):
        lines.append(f"{LETTER[i]}) {str(opt).strip()}")
    return "\n".join(lines) + "\n"

def row_to_item(row: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """
    Convert one GPQA row to MMLU-Pro-like item.
    Supports two schemas:
      1) Columnar: 'Question', 'Correct Answer', 'Incorrect Answer 1..3'
      2) Key-based: 'question', 'options', 'answer' (index)
    """
    # --- schema 1: columnar (common in gpqa_main) ---
    if "Question" in row and "Correct Answer" in row:
        q = str(row.get("Question", "")).strip()
        correct = str(row.get("Correct Answer", "")).strip()
        inc1 = str(row.get("Incorrect Answer 1", "")).strip()
        inc2 = str(row.get("Incorrect Answer 2", "")).strip()
        inc3 = str(row.get("Incorrect Answer 3", "")).strip()
        # Keep the original order as A: Correct, B-D: Incorrects
        options = [opt for opt in [correct, inc1, inc2, inc3] if opt != ""]
        # Label is 0 (first option) if we keep this fixed order
        gold_idx = 0
        text = build_text(q, options)
        return {
            "task": f"GPQA ({task_name})",
            "category": "science",
            "text": text,
            "label": int(gold_idx),
            "num_options": len(options),
        }

    # --- schema 2: key-based (defensive branch) ---
    if "question" in row and "options" in row and "answer" in row:
        q = str(row.get("question", "")).strip()
        options = [str(x).strip() for x in (row.get("options") or [])]
        try:
            gold_idx = int(row.get("answer"))
        except Exception:
            gold_idx = -1
        text = build_text(q, options)
        return {
            "task": f"GPQA ({task_name})",
            "category": "science",
            "text": text,
            "label": int(gold_idx),
            "num_options": len(options),
        }

    # If neither schema is found, return an empty/invalid record
    raise ValueError(f"[ERROR] Unexpected schema in GPQA ({task_name}): {row}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gpqa_main",
                        choices=["gpqa_main","gpqa_diamond","gpqa_extended","gpqa_experts"],
                        help="GPQA configuration to load")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (most GPQA configs provide 'train')")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, export only first N samples")
    parser.add_argument("--out", type=str, default="./gpqa_mmlupro.json",
                        help="Output JSON path")
    args = parser.parse_args()

    ds = load_dataset("Idavidrein/gpqa", args.config, split=args.split)
    n = len(ds)
    take = n if args.limit <= 0 else min(args.limit, n)

    print(f"Loaded GPQA ({args.config}/{args.split}) with {n} samples. Exporting {take}.")

    export: List[Dict[str, Any]] = []
    for i in range(take):
        item = row_to_item(ds[i], task_name=args.config)
        export.append(item)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to: {args.out}")
    # Show a tiny preview
    print(json.dumps(export[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()