#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset
import json
import sys

# ---------------- Config ----------------
DATASET_NAME = "Idavidrein/gpqa"
CONFIG_NAME  = "gpqa_main"   # one of: gpqa_main, gpqa_diamond, gpqa_extended, gpqa_experts
SPLIT        = "train"        # GPQA provides test only
N_SAMPLES    = 5             # number of samples to preview
# ----------------------------------------


def main():
    try:
        ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
    except Exception as e:
        print(
            "Failed to load dataset. Make sure CONFIG_NAME is one of "
            "['gpqa_main','gpqa_diamond','gpqa_extended','gpqa_experts'].\n"
            f"Current CONFIG_NAME='{CONFIG_NAME}'.\nError: {e}"
        )
        sys.exit(1)

    print(f"Loaded GPQA ({CONFIG_NAME}/{SPLIT}) with {len(ds)} samples.")

    samples = []
    take = min(N_SAMPLES, len(ds))
    for i in range(take):
        row = ds[i]

        # Expected fields in GPQA
        question = row.get("question", "")
        options  = list(row.get("options", []))
        # 'answer' is an integer index in GPQA
        answer_idx = int(row.get("answer", -1))

        # Build text (MMLU-Pro style format)
        text = question.strip()
        for j, opt in enumerate(options):
            text += f"\n{chr(ord('A') + j)}) {opt}"
        text += "\n"

        item = {
            "task": f"GPQA ({CONFIG_NAME})",
            "category": "science",
            "text": text,
            "label": answer_idx,
            "num_options": len(options),
        }
        samples.append(item)

    # Pretty print
    print(json.dumps(samples, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()