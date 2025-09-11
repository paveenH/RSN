#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:38:16 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download LogiQA2.0 MRC (English) train/dev/test and merge into ONE JSON:
{
  "data": [
    {
      "task": "MRC"  # or "MRC-train/dev/test" if ADD_SPLIT_TO_TASK=True
      "text": "<passage>\n\n<question>\nA) ...\nB) ...\nC) ...\nD) ...\n",
      "label": <0-based correct index>,
      "num_options": 4
    },
    ...
  ]
}

- Deterministic option shuffling (SEED).
- Input files are JSON Lines from the provided GitHub repo.
"""

import os
import json
import urllib.request
import random
from typing import List, Dict, Any

# ========== CONFIG ==========
SAVE_DIR = "/data2/paveen/RolePlaying/components/logiqa"
OUT_PATH = os.path.join(SAVE_DIR, "logiqa_mrc.json")

# GitHub (csitfun/LogiQA2.0) → use RAW URLs
URLS = {
    "train": "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/train.txt",
    "dev":   "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/dev.txt",
    "test":  "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA/test.txt",
}

# shuffle options?
SHUFFLE_OPTIONS = False
SEED = 42

# include split info in task? (e.g., "MRC-train/dev/test")
ADD_SPLIT_TO_TASK = True
# ===========================

LETTER = [chr(ord("A") + i) for i in range(26)]
rnd = random.Random(SEED)

def _download_lines(url: str) -> List[str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        text = resp.read().decode(charset, errors="replace")
    # files are JSON Lines
    return [ln for ln in text.splitlines() if ln.strip()]

def _build_text(passage: str, question: str, options: List[str]) -> str:
    lines: List[str] = []
    passage = (passage or "").strip()
    question = (question or "").strip()
    if passage:
        lines.append(passage)
        lines.append("")  # blank
    lines.append(question)
    for i, opt in enumerate(options):
        lines.append(f"{LETTER[i]}) {str(opt).strip()}")
    return "\n".join(lines) + "\n"

def _append_item(buf: List[Dict[str, Any]], task_name: str, text: str, label: int, num_options: int):
    buf.append({
        "task": task_name,
        "text": text,
        "label": int(label),
        "num_options": int(num_options),
    })

def _coerce_gold(gold: Any, nopt: int) -> int | None:
    """
    gold should be 0-based int. Some files may use str digits; coerce safely.
    """
    if isinstance(gold, int):
        return gold if 0 <= gold < nopt else None
    try:
        gi = int(gold)
        return gi if 0 <= gi < nopt else None
    except Exception:
        return None

def _load_split(split: str, url: str) -> List[Dict[str, Any]]:
    buf: List[Dict[str, Any]] = []
    task_base = "MRC"
    task_name = f"{task_base}-{split}" if ADD_SPLIT_TO_TASK else task_base

    for ln in _download_lines(url):
        ex = json.loads(ln)
        passage = ex.get("text", "")
        question = ex.get("question", "")
        options_raw = ex.get("options") or []
        options = [str(x).strip() for x in options_raw if str(x).strip()]
        if not question or len(options) < 2:
            continue

        gold = _coerce_gold(ex.get("answer"), len(options))
        if gold is None:
            continue

        if SHUFFLE_OPTIONS:
            perm = list(range(len(options)))
            rnd.shuffle(perm)
            options_shuf = [options[j] for j in perm]
            label = perm.index(gold)
        else:
            options_shuf = options
            label = gold

        text = _build_text(passage, question, options_shuf)
        _append_item(buf, task_name, text, label, len(options_shuf))
    return buf

def main():
    merged: List[Dict[str, Any]] = []
    per_split = {}

    for split, url in URLS.items():
        items = _load_split(split, url)
        merged.extend(items)
        per_split[split] = len(items)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"data": merged}, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved merged LogiQA2.0 MRC (EN) → {OUT_PATH}")
    total = sum(per_split.values())
    print(f"[INFO] Total items: {total} | " + " | ".join(f"{k}:{v}" for k, v in per_split.items()))

if __name__ == "__main__":
    main()