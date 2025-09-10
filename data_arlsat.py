#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AR-LSAT (local JSON) → MMLU-Pro-like MCQ JSON.

- Reads local JSON files (downloaded from the AR-LSAT GitHub repo).
- Converts each question under a passage into:
    {
      "task": "AR-LSAT",
      "category": "law",
      "text": "<passage>\n\n<question>\nA) ...\nB) ...\n...",
      "label": <0-based correct index>,
      "num_options": K
    }
- Optional: shuffle options (default False).
- Supports 4- or 5-choice items.
- Supports answers in letter / index / text format.
"""

import os
import json
import glob
import random
from typing import List, Dict, Any

# ========== CONFIG ==========
DATA_DIR   = "/data2/paveen/RolePlaying/datasets/AR-LSAT/complete_lsat_data"  # local directory
# Will automatically read *.json files (e.g., train.json / dev.json / test.json)

SAVE_DIR   = "/data2/paveen/RolePlaying/components/arlsat"
OUT_PATH   = os.path.join(SAVE_DIR, "arlsat_all.json")

SHUFFLE_OPTIONS = True       # Whether to shuffle answer options
SEED            = 42         # Random seed for reproducibility
# ===========================

LETTER = [chr(ord("A")+i) for i in range(26)]  # A..Z

rnd = random.Random(SEED)

def _read_all_json(data_dir: str) -> List[Dict[str, Any]]:
    """Read all *.json files under data_dir and concatenate top-level lists."""
    all_items: List[Dict[str, Any]] = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No JSON files found under: {data_dir}")
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Top-level JSON must be a list: {p}")
        all_items.extend(data)
    return all_items

def _letter_to_index(ans: str) -> int:
    s = (ans or "").strip().upper()
    if len(s) == 1 and "A" <= s <= "Z":
        return ord(s) - ord("A")
    return -1

def _build_text(passage: str, question: str, options: List[str]) -> str:
    passage = (passage or "").strip()
    question = (question or "").strip()
    lines: List[str] = []
    if passage:
        lines.append(passage)
        lines.append("")  # blank line
    if question:
        lines.append(question)
    for i, opt in enumerate(options):
        lines.append(f"{LETTER[i]}) {str(opt).strip()}")
    return "\n".join(lines) + "\n"

def _resolve_gold(options: List[str], answer_raw: Any) -> int:
    """
    Map raw answer into 0-based index:
    - If it's a letter: A→0, B→1, ...
    - If it's an int / digit string: interpret as 0-based or 1-based index
    - If it's a string that exactly matches an option: take that index
    """
    # Letter
    if isinstance(answer_raw, str):
        li = _letter_to_index(answer_raw)
        if 0 <= li < len(options):
            return li

    # Integer / string of digits
    try:
        num = int(answer_raw)
        if 0 <= num < len(options):
            return num
        if 1 <= num <= len(options):
            return num - 1
    except Exception:
        pass

    # Exact text match
    if isinstance(answer_raw, str):
        s = answer_raw.strip()
        for i, opt in enumerate(options):
            if s == str(opt).strip():
                return i

    return -1

def _question_to_mc_items(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand one passage node into multiple MC items.
    Expected fields:
      - entry["passage"]: str
      - entry["questions"]: list of dicts with keys:
            ["question", "options", "answer"]
    """
    out: List[Dict[str, Any]] = []
    passage = entry.get("passage", "")
    qlist   = entry.get("questions", []) or []
    if not isinstance(qlist, list):
        return out

    for qnode in qlist:
        q_text = qnode.get("question", "")
        opts   = qnode.get("options", []) or []
        options = [str(x).strip() for x in opts if str(x).strip()]

        if not q_text or not options:
            continue

        gold = _resolve_gold(options, qnode.get("answer", None))
        if gold < 0:
            continue

        if SHUFFLE_OPTIONS:
            perm = list(range(len(options)))
            rnd.shuffle(perm)
            options_shuf = [options[j] for j in perm]
            label = perm.index(gold)
        else:
            options_shuf = options
            label = gold

        item = {
            "task": "AR-LSAT",
            "category": "law",
            "text": _build_text(passage, q_text, options_shuf),
            "label": int(label),
            "num_options": len(options_shuf),
        }
        out.append(item)

    return out

def main():
    # 1) Read all JSON files
    entries = _read_all_json(DATA_DIR)
    print(f"[LOAD] Found {len(entries)} top-level passages under: {DATA_DIR}")

    # 2) Convert to MC items
    merged: List[Dict[str, Any]] = []
    for entry in entries:
        merged.extend(_question_to_mc_items(entry))

    # 3) Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved AR-LSAT MCQ → {OUT_PATH}")
    print(f"[INFO] Total MC items: {len(merged)}")
    if merged:
        print("[Preview top-2]")
        print(json.dumps(merged[:2], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()