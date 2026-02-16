#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and format GSM8K dataset for open-ended generation evaluation.

GSM8K is a math word problem dataset. Unlike MMLU (multiple-choice),
answers are free-form numbers extracted from the solution text.

Output JSON format:
[
  {
    "task": "gsm8k",
    "question": "Natalia sold clips to ...",
    "answer": "72",           # ground truth numeric answer
    "solution": "Step 1 ... #### 72"  # full CoT solution
  },
  ...
]

@author: paveenhuang
"""

import os
import json
import re
from datasets import load_dataset


def extract_answer(solution: str) -> str:
    """
    GSM8K solutions end with '#### <number>'.
    Extract the final numeric answer.
    """
    match = re.search(r"####\s*(.+)", solution)
    if match:
        # Remove commas from numbers like "1,234"
        return match.group(1).strip().replace(",", "")
    return ""


if __name__ == "__main__":
    # -------- Paths --------
    cache_dir = "/data1/paveen/RolePlaying/.cache"
    save_dir = "/data1/paveen/RolePlaying/components/benchmark"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Load GSM8K --------
    # GSM8K has "train" and "test" splits
    for split in ["test"]:
        ds = load_dataset(
            "openai/gsm8k",
            "main",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        export = []
        for row in ds:
            question = row["question"]
            solution = row["answer"]  # full CoT solution with #### at end
            answer = extract_answer(solution)

            export.append({
                "task": "gsm8k",
                "question": question,
                "answer": answer,
                "solution": solution,
            })

        out_path = os.path.join(save_dir, f"gsm8k_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        print(f"Saved GSM8K ({split}): {len(export)} samples -> {out_path}")
