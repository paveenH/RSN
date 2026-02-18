#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:43:50 2026

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove repetitive trailing content from LLM-generated GSM8K responses.

Detects and removes repeated phrases/sentences at the end of generated text,
keeping only the first meaningful occurrence. Specifically optimized for 
Qwen/Llama hallucinations, but uses a SOFT CLEAN strategy to preserve 
identity override / persona drift texts (e.g., "I am a human who...").

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
    温和版清洗策略（Soft Clean）：
    旨在保留模型因 Steering 产生的身份幻觉（如 "I am a human"），
    只切除纯粹的 token 死循环和机械性的标点/数字无限重复。
    """
    if not text or len(text) < 10:
        return text

    # ─── Step 1: 仅拦截疯狂连续的 #### 死循环 ───
    # 如果它连续输出了 3 次以上的 ####数字，才判定为死循环截断。
    # 像 ####90####. 这种偶尔多写一次的，放过它。
    if len(re.findall(r'####\s*[-+]?[\d.,]+', text)) >= 3:
        first_marker = re.search(r'####\s*[-+]?[\d.,]+', text)
        if first_marker:
            text = text[:first_marker.end()].strip()

    # ─── Step 2: 拦截确切的块状死循环（纯机械重复） ───
    # 匹配完全一模一样的 15 个字符以上内容，连续重复 3 次以上
    loop_pattern = re.compile(r'(?s)(.{15,}?)(?:\s*\1){3,}')
    while True:
        match = loop_pattern.search(text)
        if not match:
            break
        # 保留到第一次重复结束
        cut_idx = match.start(1) + len(match.group(1))
        text = text[:cut_idx].strip()

    # ─── Step 3: 拦截短词/数字死循环 (如 450. 450. 450. 450.) ───
    short_loop = re.compile(r'((?:\b\S+\b\s*[.,!?]*\s*){1,5}?)(?:\1){4,}')
    while True:
        match = short_loop.search(text)
        if not match:
            break
        cut_idx = match.start(1) + len(match.group(1))
        text = text[:cut_idx].strip()

    # ─── Step 4: 移除原来会误伤 "I hope it is correct" 的 halt_phrases 拦截 ───
    # 仅保留完全一模一样的句子连续重复 3 次以上的硬截断
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    if len(sentences) > 3:
        cleaned_sentences = []
        consecutive_repeats = 0
        last_norm = ""

        for s in sentences:
            norm = re.sub(r'[^a-z0-9]', '', s.lower())
            
            # Skip very short fragments
            if len(norm) < 8:
                cleaned_sentences.append(s)
                continue

            # 只拦截完全一模一样的相邻句子重复（容忍递进扩写）
            if norm == last_norm:
                consecutive_repeats += 1
                if consecutive_repeats >= 2: # 连续出现 3 次一样的句子才截断
                    break 
            else:
                consecutive_repeats = 0
            
            last_norm = norm
            cleaned_sentences.append(s)

        text = " ".join(cleaned_sentences).strip()

    # ─── Step 5: 截断超长文本（防止炸显存） ───
    # 适当放宽长度，给模型表演身份崩溃的空间
    if len(text) > 3000:
        text = text[:3000] + " ... [TRUNCATED]"

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