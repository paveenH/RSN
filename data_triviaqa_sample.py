#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and format TriviaQA dataset (300-sample version) for faster evaluation.

Samples 300 examples from the validation set for quick iteration.

Output JSON format:
[
  {
    "task": "triviaqa",
    "question": "Who was the first president of ...",
    "answer": "George Washington",
    "aliases": ["Washington", "george washington", ...]
  },
  ...
]

@author: paveenhuang
"""

import os
import json
import random
from datasets import load_dataset


if __name__ == "__main__":
    # -------- Configuration --------
    SAMPLE_SIZE = 300
    RANDOM_SEED = 42

    # -------- Paths --------
    cache_dir = "/data1/paveen/RolePlaying/.cache"
    save_dir = "/data1/paveen/RolePlaying/components/benchmark"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Load TriviaQA --------
    # TriviaQA "rc.nocontext" subset: question-answer only (no document context)
    # Splits: train, validation, test (test has no answers)
    for split in ["validation"]:
        ds = load_dataset(
            "trivia_qa",
            "rc.nocontext",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Sample 300 examples
        random.seed(RANDOM_SEED)
        total_samples = len(ds)
        if total_samples <= SAMPLE_SIZE:
            # If dataset is smaller than sample size, use all
            indices = list(range(total_samples))
        else:
            # Random sample without replacement
            indices = random.sample(range(total_samples), SAMPLE_SIZE)
            indices.sort()  # Sort for reproducibility

        export = []
        for idx in indices:
            row = ds[idx]
            question = row["question"]
            answer_obj = row["answer"]
            # answer_obj has: "value" (canonical), "aliases" (list of acceptable answers)
            answer = answer_obj["value"]
            aliases = answer_obj.get("aliases", [])

            export.append({
                "task": "triviaqa",
                "question": question,
                "answer": answer,
                "aliases": aliases,
            })

        out_path = os.path.join(save_dir, f"triviaqa_{split}_sample.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        print(f"Saved TriviaQA ({split}) sample: {len(export)}/{total_samples} samples -> {out_path}")
        print(f"  Random seed: {RANDOM_SEED}")
