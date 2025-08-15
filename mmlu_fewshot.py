#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_mmlu_fewshot.py (validation-based)

From the HuggingFace `lukaemon/mmlu` dataset, deterministically sample k
VALIDATION examples per task (subject) and save them as local JSON files that
are directly compatible with the few-shot constructor (`build_fewshot_prefix`).

Each saved JSON contains a list of objects with fields:
- id:     a stable identifier like "{task}_val_{index}"
- task:   subject name (space-separated, e.g., "abstract algebra")
- text:   the question stem (dataset["input"])
- choices: ["A_text", "B_text", "C_text", "D_text"]
- label:  integer in [0..3], where 0->A, 1->B, 2->C, 3->D

Usage:
    python prepare_mmlu_fewshot.py \
        --save_dir /path/to/mmlu_fewshot \
        --cache_dir /path/to/hf_cache \
        --k 5 \
        --global_seed 0
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import List

from datasets import load_dataset
from detection.task_list import TASKS


def stable_seed(*parts, global_seed: int = 0) -> int:
    """Deterministically derive a 32-bit seed from stringable parts + global_seed."""
    h = hashlib.sha256(("||".join(map(str, parts)) + f"||{global_seed}").encode()).hexdigest()
    return int(h[:8], 16)  # first 32 bits


def pick_k_indices(n: int, k: int, task: str, split: str, global_seed: int) -> List[int]:
    rng = random.Random(stable_seed(task, split, global_seed))
    idxs = list(range(n))
    rng.shuffle(idxs)
    return idxs[:k] if k < n else idxs


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir = "/data2/paveen/RolePlaying/src/models/components/mmlu_fewshot"
    k = 5
    global_seed = 25
    overwrite = True

    os.makedirs(save_dir, exist_ok=True)

    split = "validation"

    for task in TASKS:
        out_path = os.path.join(save_dir, f"{task}.json")
        if (not overwrite) and os.path.exists(out_path):
            print(f"[Skip] {task}: {out_path} already exists. Use --overwrite to regenerate.")
            continue

        print(f"=== Loading {split} split for task: {task} ===")
        ds = load_dataset(
            "lukaemon/mmlu",
            task,
            split=split,
            cache_dir=cache_dir if cache_dir else None,
            trust_remote_code=True,
        )

        n = len(ds)
        if n == 0:
            print(f"[Warn] {split} split empty for {task}, skipping.")
            continue

        idxs = pick_k_indices(n=n, k=k, task=task, split=split, global_seed=global_seed)
        print(f"Selected {len(idxs)} / {n} {split} examples for {task} (k={k}).")

        out_items = []
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        for i in idxs:
            row = ds[i]
            q = row["input"]
            A = row["A"]
            B = row["B"]
            C = row["C"]
            D = row["D"]
            tgt = row["target"]  # "A"|"B"|"C"|"D"
            lab = label_map.get(tgt, 0)

            out_items.append(
                {
                    "id": f"{task}_val_{i}",
                    "task": task.replace("_", " "),
                    "text": q,
                    "choices": [A, B, C, D],
                    "label": lab,
                }
            )

        save_json(out_path, out_items)
        print(f"[Saved] {task}: {out_path} ({len(out_items)} items)")

    print("\n✅  All tasks processed.\n")
