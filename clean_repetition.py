#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove repetitive trailing content from LLM-generated GSM8K responses.

Detects and removes repeated phrases/sentences at the end of generated text,
keeping only the first meaningful occurrence. Specifically optimized for 
Qwen/Llama hallucinations (e.g., infinite #### loops, apology templates).

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
    Uses a multi-pass strategy to catch different types of LLM degeneration.
    """
    if not text or len(text) < 10:
        return text

    if len(text) > 3500:
        text = text[:3500]

    # ─── Step 1: Truncate at multiple #### markers ───
    first_marker = re.search(r'####\s*[-+]?[\d.,]+', text)
    if first_marker and text.count('####') > 1:
        text = text[:first_marker.end()].strip()

    # ─── Step 2: Catch exact character block loops (Sentence/Paragraph loops) ───
    loop_pattern = re.compile(r'(?s)(.{10,200}?)(?:\s*\1){2,}')
    while True:
        match = loop_pattern.search(text)
        if not match:
            break
        # Keep up to the end of the FIRST occurrence
        cut_idx = match.start(1) + len(match.group(1))
        text = text[:cut_idx].strip()

    # ─── Step 3: Catch short word/number loops ───
    short_loop = re.compile(r'((?:\b\S+\b\s*[.,!?]*\s*){1,5}?)(?:\1){3,}')
    while True:
        match = short_loop.search(text)
        if not match:
            break
        cut_idx = match.start(1) + len(match.group(1))
        text = text[:cut_idx].strip()

    # ─── Step 4: Sentence-level Deduplication & Template Hallucinations ───
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    if len(sentences) > 2:
        cleaned_sentences = []
        seen_normalized = set()
        consecutive_repeats = 0

        # Known Qwen degeneration / RLHF refusal templates
        halt_phrases = [
            "checked my work", 
            "hope it is correct", 
            "thank you for your understanding", 
            "let me know if you need", 
            "i have done my job", 
            "i am done",
            "further assistance",
            "sincerely"
        ]

        for s in sentences:
            norm = re.sub(r'[^a-z0-9]', '', s.lower())
            
            if len(norm) < 8:
                cleaned_sentences.append(s)
                continue

            if any(re.sub(r'[^a-z0-9]', '', phrase) in norm for phrase in halt_phrases):
                break

            if norm in seen_normalized:
                consecutive_repeats += 1
                if consecutive_repeats >= 2:
                    break
            else:
                consecutive_repeats = 0

            seen_normalized.add(norm)
            cleaned_sentences.append(s)

        text = " ".join(cleaned_sentences).strip()

    # ─── Step 5: Fallback Length Cutoff ───
    if len(text) > 2500:
        text = text[:2500] + " ... [TRUNCATED]"

    return text


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

        if cleaned != original:
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

    # Find all answer files (ignores already cleaned files)
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


if __name__ == "__main__":
    main()