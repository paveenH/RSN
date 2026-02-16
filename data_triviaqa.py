#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and format TriviaQA dataset for open-ended generation evaluation.

TriviaQA is a reading comprehension dataset with question-answer pairs.
Unlike MMLU (multiple-choice), answers are free-form text.

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
from datasets import load_dataset


if __name__ == "__main__":
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

        export = []
        for row in ds:
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

        out_path = os.path.join(save_dir, f"triviaqa_{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        print(f"Saved TriviaQA ({split}): {len(export)} samples -> {out_path}")
