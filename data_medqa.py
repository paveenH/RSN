#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export MedQA (English, source) to a single JSON file in MMLU-Pro-like format.

Dataset: bigbio/med_qa
Config : med_qa_en_source
Split  : test (default, can also be train/validation)

Output JSON fields per sample:
- task:        "MedQA (source)"
- category:    "medicine"
- text:        "Question...\nA) ...\nB) ...\n..."
- label:       int (0-based gold index); -1 if unavailable
- num_options: int (number of answer options)
"""

import os, json, ast
from typing import Any, List, Optional, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

LETTER10 = ["A","B","C","D","E","F","G","H","I","J"]

def _normalize_option_item(item: Any) -> str:
    """Normalize options to plain text."""
    if isinstance(item, dict):
        if "value" in item:
            return str(item["value"]).strip()
        if "text" in item:
            return str(item["text"]).strip()
        return str(item).strip()
    if isinstance(item, str):
        s = item.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    if "value" in d:
                        return str(d["value"]).strip()
                    if "text" in d:
                        return str(d["text"]).strip()
                    return str(d).strip()
            except Exception:
                pass
        return s
    return str(item).strip()

def _get_options(row: Dict[str, Any]) -> List[str]:
    opts = row.get("options", None)
    if isinstance(opts, list):
        return [_normalize_option_item(o) for o in opts]
    for k in ("choices", "options_text", "answers"):
        v = row.get(k, None)
        if isinstance(v, list):
            return [_normalize_option_item(o) for o in v]
    return []

def _letter_to_index(letter: str) -> Optional[int]:
    letter = (letter or "").strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        return ord(letter) - ord("A")
    return None

def _get_answer_idx(row: dict, options: List[str]) -> Optional[int]:
    ai = row.get("answer_idx", None)
    if ai is not None:
        if isinstance(ai, str):
            li = _letter_to_index(ai)
            if li is not None and 0 <= li < len(options):
                return li
            try:
                num = int(ai)
                if 0 <= num < len(options):
                    return num
            except Exception:
                pass
        else:
            try:
                num = int(ai)
                if 0 <= num < len(options):
                    return num
            except Exception:
                pass
    ans = row.get("answer", None)
    if ans is not None:
        ans_s = str(ans).strip()
        li = _letter_to_index(ans_s)
        if li is not None and 0 <= li < len(options):
            return li
        try:
            idx = options.index(ans_s)
            return idx
        except ValueError:
            pass
    return None

def _format_mc_text(question: str, options: List[str], letters: List[str]) -> str:
    text = question.strip()
    K = min(len(options), len(letters))
    for i in range(K):
        text += f"\n{letters[i]}) {options[i]}"
    return text + "\n"

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
        assert split in ["train", "validation", "test"], "split must be one of train/validation/test"
        self.split = split
        self.option_letters = option_letters
        self.option_separator = option_separator
        self.postfix_token = None
        if postfix_token is not None:
            self.postfix_token = torch.ones((1,), dtype=torch.long) * postfix_token
        self.dataset = load_dataset(
            "bigbio/med_qa",
            "med_qa_en_source",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[idx]
        question = row.get("question", "")
        options = _get_options(row)
        label_idx = _get_answer_idx(row, options)
        if label_idx is None:
            raise ValueError(f"[Error] Sample {idx} has no gold answer. Row={row}")
        text = _format_mc_text(question, options, self.option_letters)
        return {
            "text": text,
            "label": int(label_idx),
            "task": "MedQA (source)",
            "category": "medicine",
            "num_options": len(options),   # <-- added
        }

if __name__ == "__main__":
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/medqa"
    os.makedirs(save_dir, exist_ok=True)

    split = "test"   # "train"/"validation"

    ds = MedQASource(cache_dir=cache_dir, split=split)
    export = []
    print(f"Loaded MedQA (source, {split}) with {len(ds)} samples.")
    no_gold = 0

    for i in range(len(ds)):
        samp = ds[i]
        if samp["label"] < 0:
            no_gold += 1
        export.append({
            "task": samp["task"],
            "category": samp["category"],
            "text": samp["text"],
            "label": samp["label"],
            "num_options": samp["num_options"],   # <-- added
        })

    out_path = os.path.join(save_dir, f"medqa_source_{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved MedQA (source, {split}) to: {out_path}")
    if no_gold:
        print(f"ℹ️  {no_gold} / {len(ds)} rows have no gold label (label = -1).")