#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download & merge AR-LSAT train/dev/test into ONE JSON file.
Mark split in the "task" field: "AR-LSAT-train", "AR-LSAT-dev", "AR-LSAT-test".
"""

import os
import json
import random
import urllib.request
from typing import List, Dict, Any, Union

# ========== CONFIG ==========
# Recommended: use raw.githubusercontent.com URLs
URLS: List[str] = [
    "https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_TrainData.json",
    "https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_DevData.json",
    "https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_TestData.json",
]

SAVE_DIR = "/data2/paveen/RolePlaying/components/arlsat"
OUT_PATH = os.path.join(SAVE_DIR, "arlsat_all.json")

SHUFFLE_OPTIONS = True   # Whether to shuffle options
SEED            = 42     # Random seed
# ===========================

LETTER = [chr(ord("A") + i) for i in range(26)]
rnd = random.Random(SEED)

def _load_source(src: str) -> Any:
    """Load a JSON list from either a URL (http/https) or a local file path."""
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src, timeout=60) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            text = resp.read().decode(charset, errors="replace")
            return json.loads(text)
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_split_tag(src: str) -> str:
    """Infer split tag from source name/path."""
    s = src.lower()
    if "train" in s:
        return "train"
    if "dev" in s or "valid" in s or "val" in s:
        return "dev"
    if "test" in s:
        return "test"
    return "unknown"

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

def _resolve_gold(options: List[str], answer_raw: Union[str, int, None]) -> int:
    """
    Map the raw answer to a 0-based index:
      - If it's a letter: A→0, B→1, ...
      - If it's an int or digit string: first try 0-based, then fall back to 1-based
      - If it's a string exactly equal to an option: return that option's index
    """
    if isinstance(answer_raw, str):
        li = _letter_to_index(answer_raw)
        if 0 <= li < len(options):
            return li

    try:
        num = int(answer_raw)
        if 0 <= num < len(options):
            return num
        if 1 <= num <= len(options):
            return num - 1
    except Exception:
        pass

    if isinstance(answer_raw, str):
        s = answer_raw.strip()
        for i, opt in enumerate(options):
            if s == str(opt).strip():
                return i

    return -1

def _passage_to_items(entry: Dict[str, Any], split_tag: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    passage = entry.get("passage", "")
    qlist = entry.get("questions", []) or []
    if not isinstance(qlist, list):
        return out

    for qnode in qlist:
        q_text = qnode.get("question", "")
        raw_opts = qnode.get("options", []) or []
        options = [str(x).strip() for x in raw_opts if str(x).strip()]
        if not q_text or not options:
            continue

        gold = _resolve_gold(options, qnode.get("answer"))
        if gold < 0 or gold >= len(options):
            continue

        if SHUFFLE_OPTIONS:
            perm = list(range(len(options)))
            rnd.shuffle(perm)
            options_shuf = [options[j] for j in perm]
            label = perm.index(gold)
        else:
            options_shuf = options
            label = gold

        out.append({
            "task": f"AR-LSAT-{split_tag}",
            "category": "law",
            "text": _build_text(passage, q_text, options_shuf),
            "label": int(label),
            "num_options": len(options_shuf),
        })

    return out

def main():
    merged: List[Dict[str, Any]] = []
    per_split_count = {}

    for src in URLS:
        split = _get_split_tag(src)
        data = _load_source(src)
        if not isinstance(data, list):
            raise ValueError(f"Top-level JSON must be a list. Offending source: {src}")

        count_before = len(merged)
        for entry in data:
            merged.extend(_passage_to_items(entry, split))
        per_split_count[split] = per_split_count.get(split, 0) + (len(merged) - count_before)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved merged AR-LSAT MCQs → {OUT_PATH}")
    print(f"[INFO] Total MC items: {len(merged)}")
    for k, v in per_split_count.items():
        print(f"[INFO] {k}: {v}")

if __name__ == "__main__":
    main()