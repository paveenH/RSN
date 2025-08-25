#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA (multiple_choice) → unified JSON for your pipeline.

- Builds prompt text as: question + enumerated options ("A) ...", ...).
- Returns single-label "label" (int). For MC1/MC2 we choose the FIRST positive as the gold label
  to fit a single-label classification pipeline. (Deterministic, no shuffle.)
- Keeps "category" if available.
- Saves one JSON per variant: truthfulqa_mc1_<split>.json and truthfulqa_mc2_<split>.json
"""

from typing import Any, List, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import os, json

LETTER10 = ["A","B","C","D","E","F","G","H","I","J"]


def _as_list_of_str(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, dict):
        for key in ("text", "texts", "answers", "choices", "target", "targets"):
            if key in x and isinstance(x[key], (list, tuple)):
                return [str(v) for v in x[key]]
        try:
            return [str(v) for v in x.values()]
        except Exception:
            return [str(x)]
    # 其它标量
    return [str(x)]

def _pick_one_gold_and_negatives(
    positives: List[str],
    negatives: List[str],
    max_choices: int = 10,
) -> Tuple[List[str], int]:
    """
    Deterministically build options: [first_positive] + first (max_choices-1) negatives.
    Returns (options, gold_index=0).
    """
    gold = positives[0] if positives else ""
    opts = [gold] + (negatives[: max(0, max_choices - 1)] if negatives else [])
    # fallback: if no positives, just use negatives with gold at -1 (shouldn't happen)
    if gold == "":
        opts = negatives[: max_choices]
        gold_idx = -1
    else:
        gold_idx = 0
    return opts, gold_idx


def _format_mc_text(question: str, options: List[str], letters: List[str]) -> str:
    text = question.strip()
    K = min(len(options), len(letters))
    for i in range(K):
        text += f"\n{letters[i]}) {options[i]}"
    return text + "\n"


class TruthfulQAMC(Dataset):
    """
    TruthfulQA multiple_choice wrapper:
      mode = "mc1" uses fields: mc1_targets (positives), mc1_negatives (negatives)
      mode = "mc2" uses fields: mc2_targets (positives), mc2_negatives (negatives)
    """
    def __init__(
        self,
        cache_dir: str,
        split: str = "validation",
        mode: str = "mc1",               # "mc1" or "mc2"
        option_letters: List[str] = LETTER10,
        option_separator: str = ")",
        postfix_token: int = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "validation", "test"], "split must be one of train/validation/test"
        assert mode in ["mc1", "mc2"], "mode must be 'mc1' or 'mc2'"

        self.split = split
        self.mode = mode
        self.option_letters = option_letters
        self.option_separator = option_separator

        self.postfix_token = None
        if postfix_token is not None:
            self.postfix_token = torch.ones((1,), dtype=torch.long) * postfix_token

        # Try common dataset ids; prefer canonical if available
        ds = None
        tried = []
        for repo in ["truthful_qa", "EleutherAI/truthful_qa"]:
            try:
                ds = load_dataset(repo, "multiple_choice", split=split, cache_dir=cache_dir, trust_remote_code=True)
                break
            except Exception as e:
                tried.append((repo, str(e)))
                ds = None
        if ds is None:
            raise RuntimeError(f"Failed to load TruthfulQA multiple_choice with split={split}. Tried: {tried}")
        self.dataset = ds

        # field names per mode
        if mode == "mc1":
            self.pos_key = "mc1_targets"
            self.neg_key = "mc1_negatives"
            self.task_name = "TruthfulQA MC1"
        else:
            self.pos_key = "mc2_targets"
            self.neg_key = "mc2_negatives"
            self.task_name = "TruthfulQA MC2"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        row = self.dataset[index]
        question: str = row["question"]
        positives: List[str] = row.get(self.pos_key, []) or []
        negatives: List[str] = row.get(self.neg_key, []) or []
        category: str = row.get("category", "") or ""

        if not positives:
            # Very rare, but guard anyway
            raise ValueError(f"No positive answers found at index {index} for mode={self.mode}")

        options, gold_idx = _pick_one_gold_and_negatives(positives, negatives, max_choices=len(self.option_letters))
        text = _format_mc_text(question, options, self.option_letters)

        return {
            "task": self.task_name,   # keep a task name for grouping
            "category": category,     # keep category if present
            "text": text,             # question + A..J choices
            "label": int(gold_idx),   # single gold index (0 = first positive)
        }


if __name__ == "__main__":
    # -------- Paths --------
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/truthfulqa"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Split to export --------
    split = "validation"   

    # -------- Dump MC1 --------
    ds1 = TruthfulQAMC(cache_dir=cache_dir, split=split, mode="mc1")
    export1 = []
    print(f"Loaded TruthfulQA (MC1, {split}) with {len(ds1)} samples.")
    for i in range(len(ds1)):
        s = ds1[i]
        export1.append({
            "task": s["task"],
            "category": s["category"],
            "text": s["text"],
            "label": s["label"],
        })
    out1 = os.path.join(save_dir, f"truthfulqa_mc1_{split}.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(export1, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved TruthfulQA MC1 → {out1}")

    # -------- Dump MC2 --------
    ds2 = TruthfulQAMC(cache_dir=cache_dir, split=split, mode="mc2")
    export2 = []
    print(f"Loaded TruthfulQA (MC2, {split}) with {len(ds2)} samples.")
    for i in range(len(ds2)):
        s = ds2[i]
        export2.append({
            "task": s["task"],
            "category": s["category"],
            "text": s["text"],
            "label": s["label"],
        })
    out2 = os.path.join(save_dir, f"truthfulqa_mc2_{split}.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(export2, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved TruthfulQA MC2 → {out2}")