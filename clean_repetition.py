#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove repetitive trailing content from LLM-generated GSM8K responses.

Detects and removes repeated phrases/sentences at the end of generated text,
keeping only the first meaningful occurrence.

Input:  gsm8k/*_answers_*.json
Output: gsm8k/*_answers_*_clean.json

Usage:
    python clean_repetition.py
    python clean_repetition.py --input_dir gsm8k --dry_run
"""

import os
import re
import json
import argparse


def remove_repetition(text: str) -> str:
    """
    Remove trailing repetitive content from generated text.

    Strategy: Split text into sentences, detect when sentences start repeating,
    and cut at the first repetition boundary.
    """
    if not text or len(text) < 30:
        return text

    # ─── Step 1: Handle ####N... repeated patterns first ───
    # e.g. "####3####3####3" or "#### 260. ###### 260. ######"
    m = re.search(r'####\s*[\d.,]+', text)
    if m:
        after_first_marker = m.end()
        rest = text[after_first_marker:]
        # Check if there are more #### markers after the first one
        if re.search(r'####', rest):
            # Keep everything up to and including the first #### marker
            text = text[:after_first_marker].strip()

    # ─── Step 2: Sentence-level deduplication ───
    # Split into sentences (by ., !, ?, or newline)
    # Keep sentence boundaries for clean cutting
    parts = re.split(r'(?<=[.!?\n])\s*', text)
    if len(parts) <= 2:
        return text.strip()

    # Normalize a sentence for comparison
    def normalize(s):
        s = s.strip().lower()
        s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # Track seen sentences
    seen = {}
    cut_idx = len(text)
    consecutive_repeats = 0

    for i, part in enumerate(parts):
        norm = normalize(part)
        if len(norm) < 5:
            continue

        if norm in seen:
            consecutive_repeats += 1
            if consecutive_repeats >= 2:
                # Found repeated content - cut at the position of first repeat
                first_repeat_part_idx = seen[norm]
                # Reconstruct position: join parts up to first_repeat_part_idx + 1
                kept = " ".join(parts[:first_repeat_part_idx + 1])
                cut_idx = min(cut_idx, len(kept))
                break
        else:
            consecutive_repeats = 0

        seen[norm] = i

    result = text[:cut_idx].strip()

    # ─── Step 3: Catch remaining repeated short phrases ───
    # e.g. "The answer is 3. The answer is 3. The answer is 3."
    # Use a sliding window approach on the cleaned result
    for phrase_len in range(80, 7, -1):
        if len(result) < phrase_len * 3:
            continue

        # Check if the tail is just a repeated phrase
        tail = result[-(phrase_len * 3):]
        for start in range(len(tail) - phrase_len * 2):
            chunk = tail[start:start + phrase_len]
            norm_chunk = chunk.strip()
            if len(norm_chunk) < 8:
                continue

            # Count how many times this chunk appears in the full result
            count = result.count(norm_chunk)
            if count >= 3:
                # Keep up to end of second occurrence (for context)
                first = result.find(norm_chunk)
                end_first = first + len(norm_chunk)

                # Find a good sentence boundary near end_first
                dot_pos = result.find(".", end_first - 1, end_first + 80)
                if dot_pos >= 0:
                    result = result[:dot_pos + 1].strip()
                else:
                    result = result[:end_first].strip()
                break
        else:
            continue
        break

    return result


def process_file(input_path: str, output_path: str, dry_run: bool = False) -> dict:
    """Process a single JSON file, cleaning repetitions from generated text."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["data"]
    stats = {"total": len(samples), "cleaned": 0, "chars_removed": 0}

    for sample in samples:
        original = sample.get("generated_neutral", "")
        if not original:
            continue

        cleaned = remove_repetition(original)

        if len(cleaned) < len(original):
            stats["cleaned"] += 1
            stats["chars_removed"] += len(original) - len(cleaned)
            sample["generated_neutral"] = cleaned
            sample["original_length"] = len(original)
            sample["cleaned_length"] = len(cleaned)

    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean repetitive content from GSM8K responses")
    parser.add_argument("--input_dir", type=str, default="gsm8k",
                        help="Directory containing answer JSON files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print stats, don't write files")
    args = parser.parse_args()

    # Find all answer files
    files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith(".json") and "answers" in f and "_clean" not in f
    ])

    if not files:
        print(f"No answer files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {f}")
    print()

    for fname in files:
        input_path = os.path.join(args.input_dir, fname)
        output_name = fname.replace(".json", "_clean.json")
        output_path = os.path.join(args.input_dir, output_name)

        stats = process_file(input_path, output_path, args.dry_run)

        print(f"{fname}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Cleaned: {stats['cleaned']} ({stats['cleaned']/stats['total']*100:.1f}%)")
        print(f"  Chars removed: {stats['chars_removed']:,}")
        if stats['cleaned'] > 0:
            print(f"  Avg removed per cleaned: {stats['chars_removed']/stats['cleaned']:,.0f} chars")
        if not args.dry_run:
            print(f"  Saved to: {output_path}")
        print()

    # Show a few examples
    if not args.dry_run:
        print("=" * 60)
        print("Sample comparisons (first cleaned file):")
        print("=" * 60)
        first_clean = os.path.join(args.input_dir, files[0].replace(".json", "_clean.json"))
        first_orig = os.path.join(args.input_dir, files[0])

        with open(first_orig, "r", encoding="utf-8") as f:
            orig_data = json.load(f)
        with open(first_clean, "r", encoding="utf-8") as f:
            clean_data = json.load(f)

        shown = 0
        for i, (o, c) in enumerate(zip(orig_data["data"], clean_data["data"])):
            orig_text = o.get("generated_neutral", "")
            clean_text = c.get("generated_neutral", "")
            if len(orig_text) != len(clean_text):
                print(f"\n--- Sample {i} ---")
                print(f"Q: {o['question'][:80]}...")
                print(f"Original ({len(orig_text)} chars): ...{orig_text[-100:]}")
                print(f"Cleaned  ({len(clean_text)} chars): {clean_text[-100:]}")
                shown += 1
                if shown >= 3:
                    break


if __name__ == "__main__":
    main()
