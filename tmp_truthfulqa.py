#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TruthfulQA quick inspector (fixed):
- generation (MC1): prints question, correct_answers, incorrect_answers
- multiple_choice (MC2): prints question, mc1_targets.choices/labels, mc2_targets.choices/labels
- Summaries for each config
- Silently skip non-existent splits
"""

from datasets import load_dataset
from collections import Counter
from typing import List, Any
import json


def trunc(s: Any, n: int = 220) -> str:
    s = str(s)
    return (s[:n] + " …") if len(s) > n else s


def print_one_sample_generation(row: dict) -> None:
    """Pretty print one example from the 'generation' (MC1) config."""
    print("• Keys:", list(row.keys()))
    print("  - question:", trunc(row.get("question", "")))
    print("  - category:", row.get("category"))
    print("  - type:", row.get("type"))
    # MC1 truthfulness supervision lives here:
    pos = row.get("correct_answers", []) or []
    neg = row.get("incorrect_answers", []) or []
    print(f"  - correct_answers (len={len(pos)}): {pos[:3]}")
    print(f"  - incorrect_answers (len={len(neg)}): {neg[:3]}")
    # Optional human reference:
    ba = row.get("best_answer")
    if ba:
        print("  - best_answer:", trunc(ba))


def print_one_sample_mc(row: dict) -> None:
    """Pretty print one example from the 'multiple_choice' (MC2) config."""
    print("• Keys:", list(row.keys()))
    print("  - question:", trunc(row.get("question", "")))

    def safe_show(targets: Any, name: str):
        if isinstance(targets, dict):
            choices = targets.get("choices", [])
            labels = targets.get("labels", [])
            print(f"  - {name}: dict(keys={list(targets.keys())})")
            print(f"    choices(len={len(choices)}): {choices[:4]}")
            print(f"    labels (len={len(labels)}): {labels[:10]}")
        else:
            print(f"  - {name}: {type(targets).__name__} -> {trunc(targets)}")

    safe_show(row.get("mc1_targets"), "mc1_targets")
    safe_show(row.get("mc2_targets"), "mc2_targets")


def summarize_generation(rows: List[dict]) -> None:
    """Summaries for generation (MC1) config."""
    n = len(rows)
    pos_lens = Counter(len(r.get("correct_answers", []) or []) for r in rows)
    neg_lens = Counter(len(r.get("incorrect_answers", []) or []) for r in rows)
    empty_pos = sum(1 for r in rows if len(r.get("correct_answers", []) or []) == 0)
    print(f"MC1 (generation) rows: {n}")
    print(f"  correct_answers sizes: {dict(pos_lens)}")
    print(f"  incorrect_answers sizes: {dict(neg_lens)}")
    print(f"  empty correct_answers: {empty_pos}")


def summarize_multiple_choice(rows: List[dict]) -> None:
    """Summaries for multiple_choice (MC2) config."""
    n = len(rows)
    mc1_opt_lens = Counter()
    mc2_opt_lens = Counter()
    for r in rows:
        mc1 = r.get("mc1_targets", {}) or {}
        mc2 = r.get("mc2_targets", {}) or {}
        mc1_choices = mc1.get("choices", []) if isinstance(mc1, dict) else []
        mc2_choices = mc2.get("choices", []) if isinstance(mc2, dict) else []
        mc1_opt_lens[len(mc1_choices)] += 1
        mc2_opt_lens[len(mc2_choices)] += 1
    print(f"MC (multiple_choice) rows: {n}")
    print(f"  mc1_targets #choices distribution: {dict(mc1_opt_lens)}")
    print(f"  mc2_targets #choices distribution: {dict(mc2_opt_lens)}")


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

    print("\nExample row:")
    row0 = ds[0]
    if config_name == "generation":
        print_one_sample_generation(row0)
    else:
        print_one_sample_mc(row0)

    rows = [ds[i] for i in range(min(10000, len(ds)))]
    if config_name == "generation":
        summarize_generation(rows)
    else:
        summarize_multiple_choice(rows)

    # Tiny preview with nested dicts expanded to show first few choices if present
    tiny = []
    for r in rows[:3]:
        item = {"question": r.get("question", "")}
        if config_name == "generation":
            item["correct_answers"] = (r.get("correct_answers") or [])[:3]
            item["incorrect_answers"] = (r.get("incorrect_answers") or [])[:3]
        else:
            mc1 = r.get("mc1_targets", {}) or {}
            mc2 = r.get("mc2_targets", {}) or {}
            item["mc1_choices_preview"] = mc1.get("choices", [])[:3] if isinstance(mc1, dict) else []
            item["mc1_labels_preview"] = mc1.get("labels", [])[:6] if isinstance(mc1, dict) else []
            item["mc2_choices_preview"] = mc2.get("choices", [])[:3] if isinstance(mc2, dict) else []
            item["mc2_labels_preview"] = mc2.get("labels", [])[:6] if isinstance(mc2, dict) else []
        tiny.append(item)
    print("\nTiny preview (first 3 rows):")
    print(json.dumps(tiny, ensure_ascii=False, indent=2))


def main():
    # TruthfulQA has only 'validation' split for both configs on HF.
    configs = [
        ("generation", "validation"),       # MC1 supervision
        ("multiple_choice", "validation"),  # MC2 4-choice
        ("generation", "test"),             # will be skipped (no such split)
        ("multiple_choice", "test"),        # will be skipped (no such split)
    ]
    for cfg, split in configs:
        inspect_config(cfg, split)


if __name__ == "__main__":
    main()