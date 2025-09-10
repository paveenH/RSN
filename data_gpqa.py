#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPQA (main/diamond/extended) → single MMLU-Pro-like JSON (no shuffle).

- Loads three configs: gpqa_main, gpqa_diamond, gpqa_extended
- Split defaults to 'train' (these configs expose 'train' for evaluation)
- Preserves original option order; gold index is 0 for columnar schema
- Merges all items into one JSON file
- Skips 'gpqa_experts' (not QA; metadata about experts)

Usage:
  python gpqa_merge_to_mmlupro.py \
      --split train \
      --out /path/to/gpqa_all_train.json \
      --cache_dir /your/hf/cache \
      --limit 0
"""

import os
import json
from typing import List, Dict, Any
from datasets import load_dataset
import random

rnd = random.Random(42)

LETTER = [chr(ord("A")+i) for i in range(26)]
GPQA_CONFIGS = ["gpqa_main", "gpqa_diamond", "gpqa_extended"]  # merged

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
    if "Question" in row and "Correct Answer" in row:
        q = str(row.get("Question", "")).strip()
        correct = str(row.get("Correct Answer", "")).strip()
        inc1 = str(row.get("Incorrect Answer 1", "")).strip()
        inc2 = str(row.get("Incorrect Answer 2", "")).strip()
        inc3 = str(row.get("Incorrect Answer 3", "")).strip()
        options = [opt for opt in [correct, inc1, inc2, inc3] if opt != ""]
        if not q or not options:
            raise ValueError(...)

        # shuffle options
        perm = list(range(len(options)))
        rnd.shuffle(perm)
        options_shuf = [options[j] for j in perm]
        gold_idx = perm.index(0)  # 

        text = build_text(q, options_shuf)
        return {
            "task": f"GPQA ({task_name})",
            "text": text,
            "label": int(gold_idx),
            "num_options": len(options_shuf),
        }
    

    # neither schema found → hard error
    raise ValueError(f"[ERROR] Unexpected schema in GPQA ({task_name}). Keys: {list(row.keys())[:25]}")

def main():
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/gpqa"
    out_path  = os.path.join(save_dir, "gpqa_train.json")
    split = "train"

    os.makedirs(save_dir, exist_ok=True)   

    merged: List[Dict[str, Any]] = []

    for cfg in GPQA_CONFIGS:
        print(f"[LOAD] Idavidrein/gpqa :: {cfg}/{split}")
        ds = load_dataset("Idavidrein/gpqa", cfg, split=split, cache_dir=cache_dir)
        n = len(ds)
        print(f"[INFO] {cfg}: total {n}, exporting {n}")

        for i in range(n):
            item = row_to_item(ds[i], task_name=cfg)
            merged.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved merged GPQA ({', '.join(GPQA_CONFIGS)}) → {out_path}")
    print(f"[Preview top-3]\n{json.dumps(merged[:3], ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()