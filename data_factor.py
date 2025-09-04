#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FACTOR (wiki/news/expert) → Multiple-choice JSON (MMLU-Pro-like)

- Download CSVs from GitHub raw:
    wiki_factor.csv, news_factor.csv, expert_factor.csv
- Build 4-choice MC items per row:
    Options = [completion (correct), contradiction_0, contradiction_1, contradiction_2]
    Shuffle options with a fixed seed and compute the 0-based label.
- Output fields per item:
    task, category, text, label, num_options

Notes:
- Some rows may have missing contradictions; items with no distractors are skipped.
- We lightly sanitize text (trim, fix "Asnwer:" -> "Answer:").
- No argparse; configure paths in CONFIG.
"""

import os
import io
import csv
import json
import random
import urllib.request
from typing import List, Dict, Any, Tuple

# ============== CONFIG ==============
FACTOR_CSV_URLS: List[Tuple[str, str]] = [
    ("wiki",   "https://raw.githubusercontent.com/AI21Labs/factor/main/data/wiki_factor.csv"),
    ("news",   "https://raw.githubusercontent.com/AI21Labs/factor/main/data/news_factor.csv"),
    ("expert", "https://raw.githubusercontent.com/AI21Labs/factor/main/data/expert_factor.csv"),
]

# Output
save_dir = "/data2/paveen/RolePlaying/components/factor"
out_path = os.path.join(save_dir, "factor_mc.json")

# MC rendering
LETTERS = ["A","B","C","D","E","F","G","H","I","J"]  # we only use first N as needed

# Task/category naming
TASK_NAME_BY_SPLIT = {
    "wiki":   "FACTOR Wiki (MC4)",
    "news":   "FACTOR News (MC4)",
    "expert": "FACTOR Expert (MC4)",
}
CATEGORY_BY_SPLIT = {
    "wiki":   "factuality",
    "news":   "factuality",
    "expert": "factuality",
}

# Reproducibility
SEED = 42
# ====================================


# -------- Helpers --------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def fetch_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")

def read_csv_text(csv_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for r in reader:
        # normalize keys: strip spaces
        rows.append({(k.strip() if isinstance(k, str) else k): v for k, v in r.items()})
    return rows

def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    return t.replace("Asnwer:", "Answer:")

def build_stem(full_prefix: str, context: str) -> str:
    """
    Build the stem shown before options.
    Prefer using the original 'full_prefix' (often includes 'Question:' line).
    If 'Question:' not found, prepend a neutral instruction.
    """
    prefix = normalize_text(full_prefix)
    ctx = normalize_text(context)

    parts: List[str] = []
    if "Question:" in prefix:
        parts.append(prefix)
    else:
        parts.append("Question: Read the following and choose the statement best supported by the context.")
        if prefix:
            parts.append(prefix)

    if ctx:
        parts.append("Context:")
        parts.append(ctx)

    return "\n".join([p for p in parts if p])

def format_mc_text(stem: str, options: List[str]) -> str:
    stem = normalize_text(stem)
    K = min(len(options), len(LETTERS))
    lines = [stem]
    for i in range(K):
        lines.append(f"{LETTERS[i]}) {normalize_text(options[i])}")
    return "\n".join(lines) + "\n"


# -------- Core conversion --------
def rows_to_mc_items(rows: List[Dict[str, Any]], split_name: str, rnd: random.Random) -> List[Dict[str, Any]]:
    task = TASK_NAME_BY_SPLIT.get(split_name, f"FACTOR {split_name.title()} (MC4)")
    category = CATEGORY_BY_SPLIT.get(split_name, "factuality")

    items: List[Dict[str, Any]] = []
    for r in rows:
        # Source fields per the repo schema
        full_prefix = r.get("full_prefix", "") or r.get("turncated_prefixes", "") or r.get("truncated_prefixes", "") or ""
        context     = r.get("context", "") or ""
        completion  = r.get("completion", "")

        # Collect up to three contradictions
        contras: List[str] = []
        for k in ("contradiction_0", "contradiction_1", "contradiction_2"):
            val = r.get(k)
            if val is not None and str(val).strip():
                contras.append(str(val).strip())

        # Need at least one distractor to form a choice set
        if not completion or len(contras) == 0:
            continue

        # Compose options: correct + up to 3 negatives
        options = [completion] + contras[:3]
        n_opts = len(options)

        # Shuffle and find label
        perm = list(range(n_opts))
        rnd.shuffle(perm)
        options_shuf = [options[i] for i in perm]
        label = perm.index(0)  # where the original correct (index 0) ended up

        stem = build_stem(full_prefix, context)
        text = format_mc_text(stem, options_shuf)

        items.append({
            "task": task,
            "category": category,
            "text": text,
            "label": int(label),
            "num_options": int(n_opts),
            # Optional debugging fields:
            # "_split": split_name,
            # "_perm": perm,
        })
    return items


def main():
    ensure_dir(save_dir)
    rnd = random.Random(SEED)

    all_items: List[Dict[str, Any]] = []
    for split_name, url in FACTOR_CSV_URLS:
        try:
            print(f"[DL] {split_name}: {url}")
            csv_text = fetch_text(url)
            rows = read_csv_text(csv_text)
            items = rows_to_mc_items(rows, split_name=split_name, rnd=rnd)
            print(f"[OK] {split_name}: {len(items)} MC items")
            all_items.extend(items)
        except Exception as e:
            print(f"[WARN] Failed for {split_name}: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {len(all_items)} items → {out_path}")


if __name__ == "__main__":
    main()