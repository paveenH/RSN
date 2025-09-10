#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset
import json

# Config
DATASET_NAME = "Idavidrein/gpqa"
SPLIT = "test"   # Recommended: use "test" split for evaluation
N_SAMPLES = 5    # Number of samples to preview

def main():
    # Load GPQA
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    print(f"Loaded GPQA ({SPLIT}) with {len(ds)} samples.")

    samples = []
    for i in range(N_SAMPLES):
        row = ds[i]
        question = row["question"]
        options = row["options"]
        answer_idx = int(row["answer"])
        
        # Build text (MMLU-Pro style format)
        text = question
        for j, opt in enumerate(options):
            text += f"\n{chr(ord('A')+j)}) {opt}"
        text += "\n"

        item = {
            "task": "GPQA",
            "category": "science",   # Can be refined into subcategories if needed
            "text": text,
            "label": answer_idx,
            "num_options": len(options),
        }
        samples.append(item)

    # Print a few formatted samples
    print(json.dumps(samples, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()