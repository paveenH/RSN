#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FACTOR (wiki/news/expert) → Multiple-choice JSON (MMLU-Pro-like), no "Question:" prefix.

- Download CSVs from GitHub raw and turn rows into MC items.
- No neutral "Question:" line is added.
- Optionally strip leading "Question:" / "Q:" from full_prefix itself.
"""

import os
import io
import csv
import json
import random
import re
import urllib.request
from typing import List, Dict, Any, Tuple

# ============== CONFIG ==============
FACTOR_CSV_URLS: List[Tuple[str, str]] = [
    ("wiki",   "https://raw.githubusercontent.com/AI21Labs/factor/main/data/wiki_factor.csv"),
    ("news",   "https://raw.githubusercontent.com/AI21Labs/factor/main/data/news_factor.csv"),
    ("expert", "https://raw.githubusercontent.com/AI21Labs/factor/main/data/expert_factor.csv"),
]

save_dir = "./components/factor"
out_path = os.path.join(save_dir, "factor_mc.json")

LETTERS = ["A","B","C","D","E","F","G","H","I","J"]

TASK_NAME_BY_SPLIT = {
    "wiki":   "FACTOR Wiki (MC4)",
    "news":   "FACTOR News (MC4)",
    "expert": "FACTOR Expert (MC4)",
}
CATEGORY_BY_SPLIT = {k: "factuality" for k in ("wiki","news","expert")}

SEED = 42

# NEW: stem-building behavior
ADD_NEUTRAL_INSTRUCTION = False              # don't add "Question: ..." fallback line
STRIP_LEADING_QUESTION_TOKEN = True          # strip leading "Question:" or "Q:" from full_prefix
QUESTION_TOKEN_RE = re.compile(r'^\s*(?:Question\s*:|Q\s*:)\s*', re.IGNORECASE)
# ====================================


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
        rows.append({(k.strip() if isinstance(k, str) else k): v for k, v in r.items()})
    return rows

def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    return t.replace("Asnwer:", "Answer:")

def strip_leading_question_token(s: str) -> str:
    return QUESTION_TOKEN_RE.sub("", s)

# def build_stem(full_prefix: str, context: str) -> str:
#     """
#     Build stem WITHOUT adding any 'Question:' line.
#     Optionally remove leading 'Question:' / 'Q:' tokens from full_prefix.
#     """
#     prefix = normalize_text(full_prefix)
#     ctx = normalize_text(context)

#     if STRIP_LEADING_QUESTION_TOKEN and prefix:
#         prefix = strip_leading_question_token(prefix)

#     parts: List[str] = []
#     if prefix:
#         parts.append(prefix)

#     if ctx:
#         parts.append("Context:")
#         parts.append(ctx)

#     return "\n".join([p for p in parts if p])


def build_stem(full_prefix: str, context: str) -> str:
    prefix = normalize_text(full_prefix)
    if STRIP_LEADING_QUESTION_TOKEN and prefix:
        prefix = strip_leading_question_token(prefix)
    return prefix

def format_mc_text(stem: str, options: List[str]) -> str:
    stem = normalize_text(stem)
    K = min(len(options), len(LETTERS))
    lines = [stem] if stem else []  # if stem empty, start directly with options
    for i in range(K):
        lines.append(f"{LETTERS[i]}) {normalize_text(options[i])}")
    return "\n".join(lines) + "\n"

def rows_to_mc_items(rows: List[Dict[str, Any]], split_name: str, rnd: random.Random) -> List[Dict[str, Any]]:
    task = TASK_NAME_BY_SPLIT.get(split_name, f"FACTOR {split_name.title()} (MC4)")
    category = CATEGORY_BY_SPLIT.get(split_name, "factuality")

    items: List[Dict[str, Any]] = []
    for r in rows:
        full_prefix = r.get("full_prefix", "") or r.get("turncated_prefixes", "") or r.get("truncated_prefixes", "") or ""
        context     = r.get("context", "") or ""
        completion  = r.get("completion", "")

        contras: List[str] = []
        for k in ("contradiction_0", "contradiction_1", "contradiction_2"):
            val = r.get(k)
            if val is not None and str(val).strip():
                contras.append(str(val).strip())

        if not completion or len(contras) == 0:
            continue

        options = [completion] + contras[:3]
        n_opts = len(options)

        perm = list(range(n_opts))
        rnd.shuffle(perm)
        options_shuf = [options[i] for i in perm]
        label = perm.index(0)

        stem = build_stem(full_prefix, context)

        # If you still want a fallback line ONLY when stem is totally empty:
        if ADD_NEUTRAL_INSTRUCTION and not stem:
            stem = "Read the following and choose the statement best supported by the context."

        text = format_mc_text(stem, options_shuf)

        items.append({
            "task": task,
            "category": category,
            "text": text,
            "label": int(label),
            "num_options": int(n_opts),
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