#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick inspector for TruthfulQA datasets.

- Loads both configs: "generation" (MC1) and "multiple_choice" (MC2)
- Inspects splits: validation and test (skip non-existent silently)
- Prints: dataset info, features, sample rows, and basic stats
- Robustly handles mc1_targets/mc2_targets as list or dict
"""

from datasets import load_dataset
from collections import Counter
from typing import List, Any
import json


def as_list_of_str(x: Any) -> List[str]:
    """Coerce various shapes to List[str] for targets fields."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, dict):
        # Prefer common keys if present
        for k in ("text", "texts", "answers", "choices", "target", "targets"):
            if k in x and isinstance(x[k], (list, tuple)):
                return [str(v) for v in x[k]]
        # Otherwise all values
        try:
            return [str(v) for v in x.values()]
        except Exception:
            return [str(x)]
    return [str(x)]


def print_one_sample(row: dict, keys: List[str], max_opt_show: int = 5) -> None:
    """Pretty print one example row with safe handling."""
    def trunc(s: str, n=220):
        s = str(s)
        return (s[:n] + " …") if len(s) > n else s

    print("• Keys:", list(row.keys()))
    for k in keys:
        if k in row:
            val = row[k]
            if isinstance(val, str):
                print(f"  - {k}: {trunc(val)}")
            elif isinstance(val, list):
                print(f"  - {k}: list(len={len(val)}) -> {trunc(val[:max_opt_show])}")
            elif isinstance(val, dict):
                print(f"  - {k}: dict(keys={list(val.keys())})")
            else:
                print(f"  - {k}: {type(val).__name__} -> {trunc(val)}")

    # MC2 options array
    if "mc2_targets" in row and isinstance(row.get("mc2_targets"), list):
        opts = row["mc2_targets"]
        print(f"  - mc2_targets (len={len(opts)}): {opts[:max_opt_show]}")
    # MC1 targets (correct/incorrect often separate)
    if "mc1_targets" in row:
        pos = as_list_of_str(row.get("mc1_targets", []))
        neg = as_list_of_str(row.get("mc1_targets_false", [])) or as_list_of_str(row.get("mc1_negatives", []))
        print(f"  - mc1_targets (pos {len(pos)}): {pos[:max_opt_show]}")
        if neg:
            print(f"  - mc1_targets_false/negatives (neg {len(neg)}): {neg[:max_opt_show]}")


def summarize_mc1(rows: List[dict]) -> None:
    """Summaries for MC1-style fields."""
    n = len(rows)
    pos_lens = Counter()
    neg_lens = Counter()
    empty_pos = 0
    for r in rows:
        pos = as_list_of_str(r.get("mc1_targets", []))
        neg = as_list_of_str(r.get("mc1_targets_false", [])) or as_list_of_str(r.get("mc1_negatives", []))
        pos_lens[len(pos)] += 1
        neg_lens[len(neg)] += 1
        if len(pos) == 0:
            empty_pos += 1
    print(f"MC1 rows: {n}")
    print(f"  Positive sizes   (count by size): {dict(pos_lens)}")
    print(f"  Negative sizes   (count by size): {dict(neg_lens)}")
    print(f"  Empty positives: {empty_pos}")


def summarize_mc2(rows: List[dict]) -> None:
    """Summaries for MC2 options."""
    n = len(rows)
    opt_lens = Counter()
    for r in rows:
        opts = r.get("mc2_targets", [])
        if isinstance(opts, dict):
            # uncommon, but handle
            opts = list(opts.values())
        opt_lens[len(opts)] += 1
    print(f"MC2 rows: {n}")
    print(f"  #options distribution: {dict(opt_lens)}")


def inspect_config(config_name: str, split: str) -> None:
    """Load and inspect one config/split."""
    print("\n" + "="*80)
    print(f"Inspecting TruthfulQA config='{config_name}', split='{split}'")
    try:
        ds = load_dataset("truthful_qa", config_name, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"  ! Failed to load: {e}")
        return

    print(f"Loaded: {ds}")
    print("Features:", ds.features)

    if len(ds) == 0:
        print("  (empty split)")
        return

    # Print one sample
    print("\nExample row:")
    print_one_sample(ds[0], keys=[
        "question", "best_answer", "correct_answers", "incorrect_answers",
        "mc1_targets", "mc1_targets_false", "mc2_targets", "category", "type"
    ])

    # Summaries
    rows = [ds[i] for i in range(min(10000, len(ds)))]  # cap for speed
    if config_name == "multiple_choice":
        summarize_mc2(rows)
    else:
        summarize_mc1(rows)

    # Dump a tiny JSON preview (first 3 rows) for quick offline inspection
    tiny = []
    for r in rows[:3]:
        tiny.append({k: (list(r[k].keys()) if isinstance(r[k], dict) else r[k]) for k in r.keys()})
    print("\nTiny preview (first 3 rows, dicts shown as their keys):")
    print(json.dumps(tiny, ensure_ascii=False, indent=2))


def main():
    # MC1 in the HF repo is usually under "generation" or "truthful_qa" default;
    # MC2 (4-choice) under "multiple_choice".
    # We’ll try the two common config names here:
    configs = [
        ("generation", "validation"),
        ("generation", "test"),
        ("multiple_choice", "validation"),
        ("multiple_choice", "test"),
    ]
    for cfg, split in configs:
        inspect_config(cfg, split)


if __name__ == "__main__":
    main()