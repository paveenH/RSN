#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze confidence-related linguistic markers in GSM8K responses.

Counts hedging words, self-correction phrases, and assertive expressions
across original / positive / negative conditions to detect confidence shifts.

Usage:
    python analyze_confidence_markers.py
    python analyze_confidence_markers.py --input_dir gsm8k --no_clean
"""

import os
import re
import json
import argparse
from collections import defaultdict


# ─────────────────── Marker definitions ───────────────────

HEDGING = [
    r"\bi think\b",
    r"\bi believe\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\bmight\b",
    r"\bcould be\b",
    r"\bpossibly\b",
    r"\bnot sure\b",
    r"\bnot certain\b",
    r"\bif i('m| am) (not )?wrong\b",
    r"\bi('m| am) not (entirely |completely )?sure\b",
    r"\bit seems\b",
    r"\bappears to\b",
    r"\bapproximately\b",
    r"\brough(ly)?\b",
]

SELF_CORRECTION = [
    r"\bwait\b",
    r"\bactually\b",
    r"\blet me reconsider\b",
    r"\blet me re-?check\b",
    r"\blet me try again\b",
    r"\blet me re-?calculate\b",
    r"\blet me re-?think\b",
    r"\bi made (a )?mistake\b",
    r"\bthat('s| is) (not right|wrong|incorrect)\b",
    r"\bsorry\b",
    r"\bcorrection\b",
    r"\bon second thought\b",
    r"\bno,\b",
    r"\bhmm\b",
    r"\boops\b",
]

ASSERTIVE = [
    r"\bclearly\b",
    r"\bobviously\b",
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bof course\b",
    r"\bwithout (a )?doubt\b",
    r"\bthe answer is\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bso the answer\b",
    r"\bwe (can )?(see|know|conclude)\b",
    r"\bit('s| is) (clear|obvious|evident)\b",
    r"\bsimpl[ey]\b",
    r"\bjust\b",
    r"\beasily\b",
]

MARKER_GROUPS = {
    "hedging": HEDGING,
    "self_correction": SELF_CORRECTION,
    "assertive": ASSERTIVE,
}


def count_markers(text, patterns):
    text_lower = text.lower()
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text_lower))
    return total


def count_markers_detail(text, patterns):
    text_lower = text.lower()
    counts = {}
    for pat in patterns:
        matches = re.findall(pat, text_lower)
        if matches:
            counts[pat] = len(matches)
    return counts


def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for sample in data["data"]:
        text = sample.get("generated_neutral", "")
        if not text:
            continue

        word_count = len(text.split())
        entry = {
            "question": sample["question"][:80],
            "correct": sample.get("correct_neutral", None),
            "text_len": len(text),
            "word_count": word_count,
        }

        for group_name, patterns in MARKER_GROUPS.items():
            raw_count = count_markers(text, patterns)
            entry[f"{group_name}_count"] = raw_count
            # Normalize per 100 words to account for response length differences
            entry[f"{group_name}_per100w"] = round(
                raw_count / max(word_count, 1) * 100, 2
            )

        results.append(entry)

    return results


def aggregate_stats(results):
    n = len(results)
    if n == 0:
        return {}

    stats = {"n": n}
    stats["avg_word_count"] = round(sum(r["word_count"] for r in results) / n, 1)

    for group_name in MARKER_GROUPS:
        counts = [r[f"{group_name}_count"] for r in results]
        per100 = [r[f"{group_name}_per100w"] for r in results]

        stats[f"{group_name}_total"] = sum(counts)
        stats[f"{group_name}_mean"] = round(sum(counts) / n, 2)
        stats[f"{group_name}_per100w_mean"] = round(sum(per100) / n, 2)
        stats[f"{group_name}_prevalence"] = round(
            sum(1 for c in counts if c > 0) / n * 100, 1
        )

    # Confidence ratio: assertive / (hedging + self_correction + 1)
    for r in results:
        r["confidence_ratio"] = round(
            r["assertive_count"]
            / (r["hedging_count"] + r["self_correction_count"] + 1),
            2,
        )
    ratios = [r["confidence_ratio"] for r in results]
    stats["confidence_ratio_mean"] = round(sum(ratios) / n, 2)

    return stats


def print_comparison(all_stats):
    conditions = list(all_stats.keys())

    print("\n" + "=" * 80)
    print("LINGUISTIC MARKER ANALYSIS - CONFIDENCE COMPARISON")
    print("=" * 80)

    header = f"{'Metric':<35}"
    for cond in conditions:
        header += f"{cond:>14}"
    print(header)
    print("-" * 80)

    metrics = [
        ("Samples", "n"),
        ("Avg word count", "avg_word_count"),
        ("", None),
        ("Hedging - total count", "hedging_total"),
        ("Hedging - mean per sample", "hedging_mean"),
        ("Hedging - per 100 words", "hedging_per100w_mean"),
        ("Hedging - % samples with any", "hedging_prevalence"),
        ("", None),
        ("Self-correction - total", "self_correction_total"),
        ("Self-correction - mean/sample", "self_correction_mean"),
        ("Self-correction - per 100 words", "self_correction_per100w_mean"),
        ("Self-correction - % with any", "self_correction_prevalence"),
        ("", None),
        ("Assertive - total count", "assertive_total"),
        ("Assertive - mean per sample", "assertive_mean"),
        ("Assertive - per 100 words", "assertive_per100w_mean"),
        ("Assertive - % samples with any", "assertive_prevalence"),
        ("", None),
        ("Confidence ratio (assert/hedge)", "confidence_ratio_mean"),
    ]

    for label, key in metrics:
        if key is None:
            print()
            continue
        row = f"{label:<35}"
        for cond in conditions:
            val = all_stats[cond].get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>14.2f}"
            else:
                row += f"{val:>14}"
        print(row)

    print("=" * 80)


def print_top_markers(filepath, condition, top_n=10):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_counts = defaultdict(int)
    for sample in data["data"]:
        text = sample.get("generated_neutral", "")
        if not text:
            continue
        for group_name, patterns in MARKER_GROUPS.items():
            detail = count_markers_detail(text, patterns)
            for pat, cnt in detail.items():
                all_counts[(group_name, pat)] += cnt

    print(f"\n  Top markers in [{condition}]:")
    sorted_markers = sorted(all_counts.items(), key=lambda x: -x[1])
    for (group, pat), cnt in sorted_markers[:top_n]:
        print(f"    {group:>16} | {pat:<40} | {cnt:>4}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze confidence markers in GSM8K responses"
    )
    parser.add_argument(
        "--input_dir", type=str, default="gsm8k",
        help="Directory containing answer JSON files",
    )
    parser.add_argument(
        "--no_clean", action="store_true",
        help="Use original files instead of _clean.json",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: input_dir/confidence_markers.csv)",
    )
    args = parser.parse_args()

    use_clean = not args.no_clean
    suffix = "_clean.json" if use_clean else ".json"

    condition_map = {"original": None, "positive": None, "negative": None}

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(suffix):
            continue
        if not use_clean and "_clean" in fname:
            continue
        for cond in condition_map:
            if cond in fname:
                condition_map[cond] = os.path.join(args.input_dir, fname)

    print("Files found:")
    for cond, path in condition_map.items():
        print(f"  {cond}: {path if path else 'NOT FOUND'}")

    found = {k: v for k, v in condition_map.items() if v is not None}
    if not found:
        print("No matching files found!")
        return

    all_stats = {}
    all_results = {}

    for cond, fpath in found.items():
        results = analyze_file(fpath)
        stats = aggregate_stats(results)
        all_stats[cond] = stats
        all_results[cond] = results

    # Overall comparison
    print_comparison(all_stats)

    # Top markers per condition
    for cond, fpath in found.items():
        print_top_markers(fpath, cond)

    # Split by correct / incorrect
    print("\n" + "=" * 80)
    print("BREAKDOWN BY CORRECTNESS")
    print("=" * 80)

    for correct_val, label in [(True, "CORRECT"), (False, "INCORRECT")]:
        sub_stats = {}
        for cond, results in all_results.items():
            sub = [r for r in results if r.get("correct") == correct_val]
            if sub:
                sub_stats[cond] = aggregate_stats(sub)
        if sub_stats:
            print(f"\n--- {label} answers ---")
            print_comparison(sub_stats)

    # Save CSV
    output_path = args.output or os.path.join(
        args.input_dir, "confidence_markers.csv"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "condition,n,avg_words,"
            "hedge_total,hedge_mean,hedge_per100w,hedge_prevalence,"
            "selfcorr_total,selfcorr_mean,selfcorr_per100w,selfcorr_prevalence,"
            "assert_total,assert_mean,assert_per100w,assert_prevalence,"
            "confidence_ratio\n"
        )
        for cond, s in all_stats.items():
            f.write(
                f"{cond},{s['n']},{s['avg_word_count']},"
                f"{s['hedging_total']},{s['hedging_mean']},{s['hedging_per100w_mean']},{s['hedging_prevalence']},"
                f"{s['self_correction_total']},{s['self_correction_mean']},{s['self_correction_per100w_mean']},{s['self_correction_prevalence']},"
                f"{s['assertive_total']},{s['assertive_mean']},{s['assertive_per100w_mean']},{s['assertive_prevalence']},"
                f"{s['confidence_ratio_mean']}\n"
            )

    print(f"\nCSV saved to: {output_path}")


if __name__ == "__main__":
    main()
