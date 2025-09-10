#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:17:52 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AR-LSAT → single MMLU-Pro-like JSON (shuffle options).

- Loads AR-LSAT from Hugging Face (tries multiple candidates).
- Shuffles options with a fixed seed; remaps gold index.
- Exports one merged JSON with items:
    { "task": "AR-LSAT",
      "text": "Question...\nA) ...\nB) ...\n...",
      "label": <0-based correct>,
      "num_options": <int> }

Notes:
- This is schema-flexible: handles (A/B/C/D + answer), (options + answer),
  or (Correct/Incorrect 1..3).
- If a passage/context field exists, it gets prefixed to the question.
- Defaults: split='train' (some releases only publish 'train' for evaluation).

Usage:
  python arlsat_to_mmlupro.py \
      --split train \
      --out /path/to/arlsat_train.json \
      --cache_dir /your/hf/cache
"""

import os
import json
import argparse
import random
from typing import Any, Dict, List, Tuple, Optional

from datasets import load_dataset

LETTER = [chr(ord("A")+i) for i in range(26)]
RND = random.Random(42)

# Try these in order
DATASET_CANDIDATES = [
    # Primary (if available)
    ("zhongwanjun/AR-LSAT", None),
    # Fallbacks (if you're using AGIEval packaging)
    # Some AGIEval mirrors expose "lsat-ar" either as a config or a subset column.
    ("TIGER-Lab/AGIEval", "lsat-ar"),
    ("lighteval/agieval", "lsat-ar"),
]

def first_nonempty(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            s = str(row[k]).strip()
            if s:
                return s
    return ""

def list_from(row: Dict[str, Any], keys: List[str]) -> List[str]:
    out = []
    for k in keys:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, (list, tuple)):
                out += [str(x).strip() for x in v if str(x).strip()]
            else:
                s = str(v).strip()
                if s:
                    out.append(s)
    return out

def collect_options_abcd(row: Dict[str, Any]) -> List[str]:
    # Common AR-LSAT/AGIEval encodings
    keys_sets = [
        ["A", "B", "C", "D", "E"],  # some games have 5 choices
        ["A", "B", "C", "D"],
        ["option_a", "option_b", "option_c", "option_d"],
        ["choice_a", "choice_b", "choice_c", "choice_d"],
    ]
    for ks in keys_sets:
        opts = []
        for k in ks:
            if k in row and row[k] is not None:
                s = str(row[k]).strip()
                if s:
                    opts.append(s)
        if len(opts) >= 2:
            return opts
    return []

def parse_answer_index(row: Dict[str, Any], n_opts: int) -> Optional[int]:
    """
    Try to parse the gold answer index from a variety of fields:
      - integer index in 'answer' / 'label'
      - letter 'A'.. in 'answer' / 'label'
      - exact string match of the 'answer_text'
    """
    cand_keys = ["answer", "label", "gold", "answer_index", "gold_index", "correct"]
    for k in cand_keys:
        if k in row and row[k] is not None:
            v = row[k]
            # numeric index?
            try:
                idx = int(v)
                if 0 <= idx < n_opts:
                    return idx
            except Exception:
                pass
            # letter?
            s = str(v).strip()
            if len(s) == 1 and "A" <= s.upper() <= "Z":
                idx = ord(s.upper()) - ord("A")
                if 0 <= idx < n_opts:
                    return idx
            # some datasets store the full correct string in 'answer'
            # we'll try to match against options in the caller (if needed)
    return None

def build_text(passage: str, question: str, options: List[str]) -> str:
    parts = []
    if passage:
        parts.append(passage.strip())
    if question:
        parts.append(question.strip())
    text = "\n".join([p for p in parts if p])
    for i, opt in enumerate(options):
        text += f"\n{LETTER[i]}) {opt.strip()}"
    return text + "\n"

def row_to_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robust extraction:
      1) Try A/B/C/D(+E) + answer/label
      2) Try options (list-like) + answer/label
      3) Try "Correct Answer" + "Incorrect Answer 1..3"
    Also tries to prepend a 'passage'/'context' if present.
    """
    # Passage/context candidates
    passage = first_nonempty(row, ["passage", "context", "setup", "story", "scenario", "content"])

    # Question candidates
    question = first_nonempty(row, ["question", "Question", "stem", "prompt", "query", "title", "text"])

    # 1) A/B/C/D(+E) style
    opts = collect_options_abcd(row)
    if opts:
        gold = parse_answer_index(row, len(opts))
        # If gold is still None and we have an answer string, try match by text
        if gold is None:
            ans_txt = first_nonempty(row, ["answer", "Answer", "correct_answer", "Correct Answer"])
            if ans_txt:
                ans_txt = ans_txt.strip()
                try:
                    gold = opts.index(ans_txt)
                except ValueError:
                    gold = None
        if gold is None:
            raise ValueError(f"[ERROR] Could not determine gold index in A/B/C/D schema. Keys: {list(row.keys())[:20]}")
        # shuffle with fixed seed
        perm = list(range(len(opts)))
        RND.shuffle(perm)
        opts_shuf = [opts[j] for j in perm]
        label = perm.index(gold)
        text = build_text(passage, question, opts_shuf)
        return {
            "task": "AR-LSAT",
            "text": text,
            "label": int(label),
            "num_options": len(opts_shuf),
        }

    # 2) options (list-like) + answer index/letter
    if "options" in row and isinstance(row["options"], (list, tuple)):
        opts = [str(x).strip() for x in row["options"] if str(x).strip()]
        if len(opts) >= 2:
            gold = parse_answer_index(row, len(opts))
            if gold is None:
                ans_txt = first_nonempty(row, ["answer", "Answer", "correct_answer", "Correct Answer"])
                if ans_txt:
                    try:
                        gold = opts.index(ans_txt.strip())
                    except ValueError:
                        gold = None
            if gold is None:
                raise ValueError(f"[ERROR] Could not determine gold index in 'options' schema. Keys: {list(row.keys())[:20]}")
            perm = list(range(len(opts)))
            RND.shuffle(perm)
            opts_shuf = [opts[j] for j in perm]
            label = perm.index(gold)
            text = build_text(passage, question, opts_shuf)
            return {
                "task": "AR-LSAT",
                "text": text,
                "label": int(label),
                "num_options": len(opts_shuf),
            }

    # 3) Correct + Incorrect 1..3
    correct = first_nonempty(row, ["Correct Answer", "correct_answer", "answer_text"])
    incorrects = list_from(row, ["Incorrect Answers", "incorrect_answers"])
    if not incorrects:
        inc_keys = [k for k in row.keys() if k.lower().startswith("incorrect answer")]
        inc_keys = sorted(inc_keys)
        incorrects = list_from(row, inc_keys)
    if correct and incorrects:
        opts = [correct] + incorrects
        perm = list(range(len(opts)))
        RND.shuffle(perm)
        opts_shuf = [opts[j] for j in perm]
        label = perm.index(0)
        text = build_text(passage, question, opts_shuf)
        return {
            "task": "AR-LSAT",
            "text": text,
            "label": int(label),
            "num_options": len(opts_shuf),
        }

    raise ValueError(f"[ERROR] Unexpected schema for AR-LSAT. Keys: {list(row.keys())[:25]}")

def try_load_any(split: str, cache_dir: Optional[str]) -> Tuple[str, Optional[str], Any]:
    last_err = None
    for ds_name, cfg in DATASET_CANDIDATES:
        try:
            ds = load_dataset(ds_name, cfg, split=split, cache_dir=cache_dir)
            return ds_name, cfg, ds
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load AR-LSAT from candidates {DATASET_CANDIDATES}: {last_err}")

def main():
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/arlsat"
    out_path  = os.path.join(save_dir, "arlsat_train.json")
    split = "train"

    ds = load_dataset("zhongwanjun/AR-LSAT", split=split, cache_dir=cache_dir)
    print(f"[LOAD] zhongwanjun/AR-LSAT :: split={split}")
    print(f"[INFO] total {len(ds)} examples")

    N = len(ds)
    export: List[Dict[str, Any]] = []
    for i in range(N):
        item = row_to_item(ds[i])
        export.append(item)

    os.makedirs(save_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved AR-LSAT → {out_path}")
    print(json.dumps(export[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()