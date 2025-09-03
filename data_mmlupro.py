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


LETTER24 = [chr(ord("A") + i) for i in range(24)]


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
        option_letters: List[str] = LETTER24,
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
        options: List[str] = row["options"]            
        category: str = row.get("category", "") or ""
        src: str = row.get("src", "") or ""

        num_options = len(options)

        # Build prompt text
        text = question
        for i in range(num_options):
            letter = self.option_letters[i]
            text += f"\n{letter}{self.option_separator} {options[i]}"
        text += "\n"

        # Label
        if "answer_index" in row and row["answer_index"] is not None:
            label_idx = int(row["answer_index"])
        else:
            ans_letter = str(row.get("answer", "")).strip()
            label_idx = int(self.target_to_idx.get(ans_letter, -1))

        # Task
        task = self._infer_task_from_src(src) or category

        return {
            "text": text,
            "label": label_idx,
            "task": task,
            "category": category,
            "num_options": num_options,
        }
    


if __name__ == "__main__":
    # -------- Paths --------
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/mmlupro"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Split to export --------
    split = "test"  # or "test" once you're ready

    # -------- Load full MMLU-Pro split --------
    ds = MMLUPro(cache_dir=cache_dir, split=split)

    # -------- Dump ALL samples into ONE json --------
    export = []
    print(f"Loaded MMLU-Pro ({split}) with {len(ds)} samples.")
    for i in range(len(ds)):
        samp = ds[i]
        export.append({
            "task": samp["task"],
            "category": samp["category"],
            "text": samp["text"],
            "label": samp["label"],
        })

    out_path = os.path.join(save_dir, f"mmlupro_{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved ALL MMLU-Pro ({split}) samples to: {out_path}")