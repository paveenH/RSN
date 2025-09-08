#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FACTOR (wiki/news/expert) → Multiple-choice JSON (MMLU-Pro-like)

Rules implemented:
- Ignore `context` entirely.
- From `full_prefix`, keep only the "question" portion:
  * Drop any trailing "Answer:" / "Asnwer:" and everything after it.
  * Strip leading "Question:" / "Q:" token.
- Clean options (completion/contradictions) by removing a leading "Answer:" / "Asnwer:" if present.
- Shuffle options deterministically and compute the 0-based correct label.
- Output fields per item: task, category, text, label, num_options.

This script only uses the Python stdlib.
"""

import os
import io
import csv
import json
import random
import re
import urllib.request
from typing import List, Dict, Any, Tuple

# =======================
# Configuration
# =======================

FACTOR_CSV_URLS: List[Tuple[str, str]] = [
    ("wiki",   "https://raw.githubusercontent.com/AI21Labs/factor/main/data/wiki_factor.csv"),
    ("news",   "https://raw.githubusercontent.com/AI21Labs/factor/main/data/news_factor.csv"),
    ("expert", "https://raw.githubusercontent.com/AI21Labs/factor/main/data/expert_factor.csv"),
]

SAVE_DIR = "./components/factor"
OUT_PATH = os.path.join(SAVE_DIR, "factor_mc.json")

LETTERS = ["A","B","C","D","E","F","G","H","I","J"]  # we only use the first N needed

TASK_NAME_BY_SPLIT = {
    "wiki":   "FACTOR Wiki (MC4)",
    "news":   "FACTOR News (MC4)",
    "expert": "FACTOR Expert (MC4)",
}
CATEGORY_BY_SPLIT = {k: "factuality" for k in ("wiki", "news", "expert")}

SEED = 42  # deterministic shuffling

# Regexes for cleaning
QUESTION_TOKEN_RE = re.compile(r'^\s*(?:Question\s*:|Q\s*:)\s*', re.IGNORECASE)
ANSWER_SPLIT_RE   = re.compile(r'(?is)\b(?:Answer|Asnwer)\s*:\s*')  # also covers the typo "Asnwer"

# =======================
# Helpers
# =======================

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
    # Normalize common typo in the dataset
    t = t.replace("Asnwer:", "Answer:")
    return t

def strip_leading_question_token(s: str) -> str:
    return QUESTION_TOKEN_RE.sub("", s)

def strip_answer_section(prefix: str) -> str:
    """
    If "Answer:" / "Asnwer:" appears, cut everything from that token to the end.
    Keeps only the question part.
    """
    m = ANSWER_SPLIT_RE.search(prefix)
    if m:
        prefix = prefix[:m.start()]
    return prefix

def squash_blank_lines(s: str) -> str:
    """Collapse multiple blank lines and trim."""
    lines = [ln.rstrip() for ln in s.splitlines()]
    out = []
    prev_blank = False
    for ln in lines:
        blank = (ln.strip() == "")
        if blank and prev_blank:
            continue
        out.append(ln)
        prev_blank = blank
    return "\n".join(out).strip()

def clean_option_text(s: str) -> str:
    """
    If an option text begins with "Answer:" / "Asnwer:", remove that token once.
    """
    s = normalize_text(s)
    s = ANSWER_SPLIT_RE.sub("", s, count=1)  # only strip the first token if present
    return s.strip()

def build_stem(full_prefix: str, context: str) -> str:
    """
    Build the stem **only from full_prefix**:
      1) Drop "Answer:" section and anything after it.
      2) Remove leading "Question:" / "Q:" token.
      3) Squash blank lines.
    Ignore `context` completely.
    """
    prefix = normalize_text(full_prefix)
    # 1) remove trailing Answer section
    prefix = strip_answer_section(prefix)
    # 2) remove leading Question/Q token
    if prefix:
        prefix = strip_leading_question_token(prefix)
    # 3) clean whitespace
    prefix = squash_blank_lines(prefix)
    return prefix

def format_mc_text(stem: str, options: List[str]) -> str:
    """
    Render stem + multiple-choice options, e.g.
      Stem text...
      A) option1
      B) option2
      ...
    """
    K = min(len(options), len(LETTERS))
    lines = [stem] if stem else []
    for i in range(K):
        lines.append(f"{LETTERS[i]}) {normalize_text(options[i])}")
    return "\n".join(lines) + "\n"

# =======================
# Core conversion
# =======================

def rows_to_mc_items(rows: List[Dict[str, Any]], split_name: str, rnd: random.Random) -> List[Dict[str, Any]]:
    task = TASK_NAME_BY_SPLIT.get(split_name, f"FACTOR {split_name.title()} (MC4)")
    category = CATEGORY_BY_SPLIT.get(split_name, "factuality")

    items: List[Dict[str, Any]] = []
    for r in rows:
        # Source fields
        full_prefix = r.get("full_prefix", "") or r.get("turncated_prefixes", "") or r.get("truncated_prefixes", "") or ""
        # We intentionally ignore `context` (often duplicates or includes answers)
        # context = r.get("context", "") or ""

        completion = clean_option_text(r.get("completion", ""))
        contras: List[str] = []
        for k in ("contradiction_0", "contradiction_1", "contradiction_2"):
            val = r.get(k)
            if val is not None and str(val).strip():
                contras.append(clean_option_text(val))

        # Need at least 1 distractor to form a choice set
        if not completion or len(contras) == 0:
            continue

        # Build options list (correct first), then shuffle
        options = [completion] + contras[:3]
        n_opts = len(options)

        perm = list(range(n_opts))
        rnd.shuffle(perm)
        options_shuf = [options[i] for i in perm]
        label = perm.index(0)  # where the original correct (index 0) ended up

        # Build stem strictly from full_prefix (question-only)
        stem = build_stem(full_prefix, context="")

        text = format_mc_text(stem, options_shuf)

        items.append({
            "task": task,
            "category": category,
            "text": text,
            "label": int(label),
            "num_options": int(n_opts),
        })

    return items

# =======================
# Main
# =======================

def main():
    ensure_dir(SAVE_DIR)
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

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {len(all_items)} items → {OUT_PATH}")

if __name__ == "__main__":
    main()