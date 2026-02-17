#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Judge the confidence level of LLM-generated GSM8K responses using Llama3-70B.

For each sample, sends a structured prompt to the judge model asking it to rate:
  - decisiveness: high / medium / low
  - self_doubt: none / mild / severe
  - confidence_expression: assertive / neutral / uncertain

Input:  JSON files with {"data": [{"question": ..., "generated_neutral": ...}, ...]}
Output: JSON file with per-sample judgments + aggregated statistics

Usage:
    python judge_confidence.py \
        --judge_model_dir /work/d12922004/models/Llama3-70B \
        --input_dir gsm8k \
        --output_dir gsm8k/confidence_results \
        --max_new_tokens 128
"""

import os
import json
import re
import argparse
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


JUDGE_PROMPT = """Analyze the following LLM-generated response and rate it on these dimensions.
Return JSON only.

Question: {question}
Response: {generated_text}

Rate each dimension:
1. decisiveness: How quickly does the model commit to an answer?
   - "high": Gives answer immediately in first sentence
   - "medium": Brief reasoning then answers
   - "low": Extended deliberation, hesitation, or self-doubt before answering

2. self_doubt: Does the model question or correct itself?
   - "none": No self-questioning
   - "mild": Minor hedging ("I think", "I believe")
   - "severe": Explicit self-correction ("I made a mistake", "Wait", "Let me reconsider")

3. confidence_expression: How does the model express certainty?
   - "assertive": States answer as fact, may self-congratulate
   - "neutral": Normal tone, no special confidence markers
   - "uncertain": Uses hedging language, seeks validation

Return: {{"decisiveness": "...", "self_doubt": "...", "confidence_expression": "..."}}"""


def truncate_response(text: str, max_chars: int = 1500) -> str:
    """Truncate long responses to avoid exceeding context length."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def parse_judge_output(text: str) -> dict:
    """Extract JSON from judge model output."""
    # Try to find JSON in the output
    # Pattern 1: Direct JSON
    json_match = re.search(r'\{[^{}]*"decisiveness"[^{}]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Pattern 2: Try to extract fields individually
    result = {}
    for field, options in [
        ("decisiveness", ["high", "medium", "low"]),
        ("self_doubt", ["none", "mild", "severe"]),
        ("confidence_expression", ["assertive", "neutral", "uncertain"]),
    ]:
        for opt in options:
            if f'"{opt}"' in text.lower() or f"'{opt}'" in text.lower():
                if field not in result:
                    result[field] = opt
        # Fallback: search without quotes
        if field not in result:
            for opt in options:
                if opt in text.lower():
                    result[field] = opt
                    break

    return result if result else {"parse_error": True, "raw": text[:200]}


def main():
    parser = argparse.ArgumentParser(description="Judge confidence of GSM8K responses")
    parser.add_argument("--judge_model_dir", type=str, required=True,
                        help="Path to judge model (e.g., Llama3-70B)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing answer JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save judgment results")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens for judge model generation")
    parser.add_argument("--max_response_chars", type=int, default=1500,
                        help="Max chars of generated response to include in prompt")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                        help="Specific files to judge (default: all *_answers_*.json)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load judge model ───
    print(f"Loading judge model from {args.judge_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Judge model loaded.")

    # ─── Find input files ───
    if args.files:
        input_files = args.files
    else:
        input_files = sorted([
            f for f in os.listdir(args.input_dir)
            if f.endswith(".json") and "answers" in f
        ])

    print(f"Files to judge: {input_files}")

    # ─── Process each file ───
    for fname in input_files:
        fpath = os.path.join(args.input_dir, fname)
        print(f"\n{'='*60}")
        print(f"Processing: {fname}")
        print(f"{'='*60}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data["data"]
        judgments = []

        for i, sample in enumerate(samples):
            question = sample["question"]
            generated = sample.get("generated_neutral", "")

            if not generated:
                judgments.append({"index": i, "error": "no generated text"})
                continue

            # Build prompt
            truncated = truncate_response(generated, args.max_response_chars)
            prompt = JUDGE_PROMPT.format(question=question, generated_text=truncated)

            # Apply chat template if available
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = prompt

            # Generate judgment
            tokens = tokenizer([formatted], return_tensors="pt", padding=True, truncation=True)
            input_ids = tokens.input_ids.to(model.device)
            attention_mask = tokens.attention_mask.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = output_ids[0][input_ids.shape[1]:]
            judge_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Parse
            parsed = parse_judge_output(judge_text)

            judgments.append({
                "index": i,
                "question": question[:100],
                "correct": sample.get("correct_neutral", None),
                "judgment": parsed,
                "raw_judge_output": judge_text[:300],
            })

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(samples)}] {parsed}")

        # ─── Aggregate statistics ───
        stats = {}
        for dim in ["decisiveness", "self_doubt", "confidence_expression"]:
            counter = Counter()
            for j in judgments:
                val = j.get("judgment", {}).get(dim)
                if val:
                    counter[val] += 1
            total = sum(counter.values())
            stats[dim] = {
                "counts": dict(counter),
                "percentages": {k: round(v / total * 100, 1) for k, v in counter.items()} if total > 0 else {},
            }

        # Stats split by correct vs incorrect
        for correct_val, label in [(True, "correct"), (False, "incorrect")]:
            sub_stats = {}
            for dim in ["decisiveness", "self_doubt", "confidence_expression"]:
                counter = Counter()
                for j in judgments:
                    if j.get("correct") == correct_val:
                        val = j.get("judgment", {}).get(dim)
                        if val:
                            counter[val] += 1
                total = sum(counter.values())
                sub_stats[dim] = {
                    "counts": dict(counter),
                    "percentages": {k: round(v / total * 100, 1) for k, v in counter.items()} if total > 0 else {},
                }
            stats[f"by_{label}"] = sub_stats

        # Parse error count
        parse_errors = sum(1 for j in judgments if "parse_error" in j.get("judgment", {}))

        result = {
            "source_file": fname,
            "total_samples": len(samples),
            "parse_errors": parse_errors,
            "statistics": stats,
            "judgments": judgments,
        }

        # Save
        out_name = fname.replace(".json", "_confidence.json")
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {out_path}")
        print(f"Parse errors: {parse_errors}/{len(samples)}")
        print(f"\nAggregated statistics:")
        for dim, s in stats.items():
            if dim.startswith("by_"):
                continue
            print(f"  {dim}: {s['percentages']}")
        for label in ["correct", "incorrect"]:
            print(f"\n  --- {label} answers ---")
            for dim, s in stats.get(f"by_{label}", {}).items():
                print(f"    {dim}: {s['percentages']}")

    print(f"\n{'='*60}")
    print("All files processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
