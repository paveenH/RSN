#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export MedQA (English, source) into MMLU-Pro style JSON.

- Builds text as: question + enumerated options ("A) ...", ...).
- Returns label as index (int).
- Uses "MedQA (source)" as the task name, and "medicine" as category.
"""

from typing import Any, List
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import os, json

LETTER10 = ["A","B","C","D","E","F","G","H","I","J"]


class MedQASource(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str = "test",
        option_letters: List[str] = LETTER10,
        option_separator: str = ")",
        postfix_token: int = None,
    ) -> None:
        super().__init__()
        assert split in ["train","validation","test"], "split must be train/validation/test"

        self.split = split
        self.option_letters = option_letters
        self.option_separator = option_separator
        self.postfix_token = (
            torch.ones((1,), dtype=torch.long) * postfix_token if postfix_token is not None else None
        )

        self.dataset = load_dataset(
            "bigbio/med_qa",
            "med_qa_en_source",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        row = self.dataset[index]
        q = row["question"]
        opts = []
        for o in row["options"]:
            # options are dicts like {"key": "A", "value": "..."}
            val = o["value"] if isinstance(o, dict) and "value" in o else str(o)
            opts.append(val)

        # Build text
        text = q
        for i, opt in enumerate(opts[:len(self.option_letters)]):
            text += f"\n{self.option_letters[i]}{self.option_separator} {opt}"
        text += "\n"

        # Label
        label_idx = None
        if row.get("answer_idx") not in (None,""):
            try:
                label_idx = int(row["answer_idx"])
            except Exception:
                label_idx = None
        if label_idx is None and row.get("answer") not in (None,""):
            try:
                label_idx = int(row["answer"])
            except Exception:
                label_idx = -1

        return {
            "text": text,
            "label": label_idx if label_idx is not None else -1,
            "task": "MedQA (source)",
            "category": "medicine",
        }


if __name__ == "__main__":
    # -------- Paths --------
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/medqa"
    os.makedirs(save_dir, exist_ok=True)

    split = "test"

    ds = MedQASource(cache_dir=cache_dir, split=split)

    export = []
    print(f"Loaded MedQA (source, {split}) with {len(ds)} samples.")
    for i in range(len(ds)):
        samp = ds[i]
        export.append({
            "task": samp["task"],
            "category": samp["category"],
            "text": samp["text"],
            "label": samp["label"],
        })

    out_path = os.path.join(save_dir, f"medqa_source_{split}.json")
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(export,f,ensure_ascii=False,indent=2)

    print(f"✅ Saved MedQA (source,{split}) samples to: {out_path}")