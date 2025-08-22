#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:58:05 2025

@author: paveenhuang
"""

from typing import Any, List, Dict
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import os, json
import re


LETTER10 = ["A","B","C","D","E","F","G","H","I","J"]


class MMLUPro(Dataset):
    """
    MMLU-Pro loader keeping interface close to your MMLU class.
    - Builds text as question + enumerated options ("A) ...", ...).
    - Returns label as index (int).
    - Adds 'category' to each item.
    - Infers a fine-grained 'task' from 'src' when possible, otherwise uses 'category'.
    """
    def __init__(
        self,
        cache_dir: str,
        split: str = "validation",
        option_letters: List[str] = LETTER10,
        option_separator: str = ")",
        postfix_token: int = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "validation", "test"], "split must be one of train/validation/test"

        self.split = split
        self.option_letters = option_letters
        self.option_separator = option_separator

        if postfix_token is not None:
            self.postfix_token = torch.ones((1,), dtype=torch.long) * postfix_token
        else:
            self.postfix_token = None

        # Load full split once (MMLU-Pro doesn't shard by subject)
        self.dataset = load_dataset(
            "TIGER-Lab/MMLU-Pro",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Build mapping A..J -> 0..9
        self.target_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.option_letters)}

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _infer_task_from_src(src: str) -> str:
        """
        Try to get a fine-grained subject from `src`.
        Example: 'cot_lib-abstract_algebra' -> 'abstract algebra'
        If not found, return empty string and let caller fallback.
        """
        if not isinstance(src, str):
            return ""
        # Heuristic: take the part after the last '-' or after 'cot_lib-'
        m = re.search(r"(?:cot_lib-)?([A-Za-z0-9_]+)$", src)
        if m:
            return m.group(1).replace("_", " ")
        return ""

    def __getitem__(self, index) -> Any:
        row = self.dataset[index]
        question: str = row["question"]
        options: List[str] = row["options"]  # list of strings
        category: str = row.get("category", "") or ""
        src: str = row.get("src", "") or ""

        # Build prompt text: question + enumerated options
        text = question
        for i, opt_text in enumerate(options):
            if i >= len(self.option_letters):
                break
            letter = self.option_letters[i]
            text += f"\n{letter}{self.option_separator} {opt_text}"
        text += "\n"

        # Label: prefer answer_index if present, else map 'answer' letter
        if "answer_index" in row and row["answer_index"] is not None:
            label_idx = int(row["answer_index"])
        else:
            # fall back to 'answer' letter like 'A'
            ans_letter = str(row["answer"]).strip()
            label_idx = int(self.target_to_idx.get(ans_letter, -1))

        # Task: try to parse from src; fallback to category
        task = self._infer_task_from_src(src)
        if not task:
            task = category

        return {
            "text": text,
            "label": label_idx,
            "task": task,        # keep MMLU-like 'task' key
            "category": category # additionally keep category as requested
        }


if __name__ == "__main__":
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/mmlupro"
    os.makedirs(save_dir, exist_ok=True)

    split = "test"
    ds = MMLUPro(cache_dir=cache_dir, split=split)
    print(f"Loaded MMLU-Pro ({split}) with {len(ds)} samples.")

    # --- group ---
    grouped = {}
    for i in range(len(ds)):
        samp = ds[i]
        task = samp["task"]
        grouped.setdefault(task, []).append({
            "task": samp["task"],
            "category": samp["category"],
            "text": samp["text"],
            "label": samp["label"],
        })

    for task, samples in grouped.items():
        fname = task.replace(" ", "_") + ".json"
        out_path = os.path.join(save_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(samples)} samples to {out_path}")

    print("\n=== All tasks have been processed and saved successfully! ===")