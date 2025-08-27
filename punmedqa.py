#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PubMedQA (pqa_labeled) → Multiple-choice (Yes/No/Maybe) JSON, MMLU-Pro-like format.

- Input dataset: qiaojin/PubMedQA, config="pqa_labeled", split="train" (only this split exists)
- Output text format:
    "Question: ...\nContext:\n<para1>\n<para2>...\nA) Yes\nB) No\nC) Maybe\n"
- Output label: yes→0, no→1, maybe→2
- Output task: "PubMedQA (labeled)"
- Output category: "medicine" (can switch to MeSH terms if desired)
"""

from typing import Any, List, Dict
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import os, json

YNM_LETTERS = ["A", "B", "C"]
YNM_TEXTS   = ["Yes", "No", "Maybe"]
DECISION_TO_IDX = {"yes": 0, "no": 1, "maybe": 2}


class PubMedQAChoice(Dataset):
    """
    Convert PubMedQA (pqa_labeled) into a 3-way multiple-choice (Yes/No/Maybe):
      - text  = Question + Context (paragraphs) + fixed options A/B/C
      - label = final_decision → 0/1/2
      - task  = "PubMedQA (labeled)"
      - category = "medicine" (can switch to ' | '.join(meshes) if desired)
    """
    def __init__(
        self,
        cache_dir: str,
        split: str = "train",                  # pqa_labeled only provides "train"
        option_letters: List[str] = YNM_LETTERS,
        option_texts:   List[str] = YNM_TEXTS,
        option_separator: str = ")",
        include_long_answer: bool = False,     # if True, also export rationale
        postfix_token: int = None,             # placeholder for interface compatibility
    ) -> None:
        super().__init__()
        assert split in ["train"], "pqa_labeled only has split='train'"

        self.split = split
        self.option_letters = option_letters
        self.option_texts = option_texts
        self.option_separator = option_separator
        self.include_long_answer = include_long_answer

        self.postfix_token = None
        if postfix_token is not None:
            self.postfix_token = torch.ones((1,), dtype=torch.long) * postfix_token

        self.dataset = load_dataset(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # A/B/C -> 0/1/2
        self.target_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.option_letters)}

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_text(self, question: str, contexts: List[str]) -> str:
        # Build Question + Context paragraphs
        parts = [f"Question: {question}", "Context:"]
        for para in contexts:
            para = (para or "").strip()
            if para:
                parts.append(para)
        # Append the fixed 3 options
        for i, opt in enumerate(self.option_texts):
            letter = self.option_letters[i]
            parts.append(f"{letter}{self.option_separator} {opt}")
        return "\n".join(parts) + "\n"

    def __getitem__(self, index) -> Any:
        row = self.dataset[index]
        question: str = row.get("question", "").strip()
        ctx_dict: dict = row.get("context", {}) or {}
        contexts: List[str] = ctx_dict.get("contexts", []) or []
        final_decision: str = (row.get("final_decision", "") or "").strip().lower()

        # label: map to 0/1/2
        if final_decision not in DECISION_TO_IDX:
            raise ValueError(f"Unexpected final_decision='{final_decision}' at index {index}")
        label_idx = DECISION_TO_IDX[final_decision]

        text = self._build_text(question, contexts)

        item = {
            "text": text,
            "label": int(label_idx),
            "task": "PubMedQA (labeled)",
            "category": "medicine",                 # or: " | ".join(meshes[:3]) for finer categories
        }
        if self.include_long_answer:
            item["long_answer"] = row.get("long_answer", "")

        return item


if __name__ == "__main__":
    # -------- Paths --------
    cache_dir = "/data2/paveen/RolePlaying/.cache"
    save_dir  = "/data2/paveen/RolePlaying/components/pubmedqa"
    os.makedirs(save_dir, exist_ok=True)

    # -------- Split (only "train" available) --------
    split = "train"

    # -------- Load and export --------
    ds = PubMedQAChoice(cache_dir=cache_dir, split=split, include_long_answer=False)

    export = []
    print(f"Loaded PubMedQA (pqa_labeled/{split}) with {len(ds)} samples.")
    for i in range(len(ds)):
        s = ds[i]
        export.append({
            "task": s["task"],
            "category": s["category"],
            "text": s["text"],
            "label": s["label"],
            # optionally include rationale:
            # "long_answer": s.get("long_answer", "")
        })

    out_path = os.path.join(save_dir, f"pubmedqa_labeled_{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved PubMedQA (pqa_labeled/{split}) to: {out_path}")