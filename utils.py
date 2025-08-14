#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:29:48 2025

@author: paveenhuang
"""

import re
import json
from typing import List, Optional


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cleaning(text: str):
    text = text.replace("<|assistant|>", "").replace("\u200b", "").strip().upper()
    m = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", text)
    return m.group(1) if m else text.strip().upper()


def make_characters(task_name: str, type_: str):
    if type_ == "none":
        task_name = task_name.replace("_", " ")
        return [
            f"none {task_name}",
            f"{task_name}",
        ]
    elif type_ == "non-":
        task_name = task_name.replace("_", "-")
        return [
            f"non-{task_name}",
            f"{task_name}",
        ]
    elif type_ == "non":
        task_name = task_name.replace("_", " ")
        return [
            f"non {task_name} expert",
            f"{task_name} expert",
            # # f"not an expert in {task_name}",
            # f"{task_name} student",
            # "person",
            "norole"
        ]
    else:
        return


def construct_prompt(vc, templates, ctx: str, role: str, use_chat: bool) -> str:
    """
    Return the final prompt string that can be fed into the model.
    - If use_chat=True: render with tokenizer.apply_chat_template (messages + add_generation_prompt=True)
    - If use_chat=False: fallback to original string templates (default/neg/neutral)
    """
    user_text = templates["vanilla"].format(context=ctx)
    # user_text = templates["neutral"].format(context=ctx)

    if role == "norole":
        messages = [{"role": "user", "content": user_text}]
        plain = templates["neutral"].format(context=ctx)
    elif "not" in role:
        messages = [
            {"role": "system", "content": f"Now you are {role}."},
            {"role": "user", "content": user_text},
        ]
        plain = templates["neg"].format(character=role, context=ctx)
    else:
        messages = [
            {"role": "system", "content": f"Now you are an honest {role}."},
            {"role": "user", "content": user_text},
        ]
        plain = templates["default"].format(character=role, context=ctx)

    if use_chat:
        return vc.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        return plain


def extract_full_correct_text(question_text: str, label_idx: int, label_map):
    prefix = f"{label_map[label_idx]})"
    for line in question_text.split("\n"):
        s = line.strip()
        if s.upper().startswith(prefix):
            return s[len(prefix) :].strip().lower()
    return None


def update(acc, char, status):
    acc[char][status] += 1


def option_token_ids(vc, LABELS):
    ids = []
    for opt in LABELS:
        tok = vc.tokenizer(opt, add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f"Option {opt} maps to {tok}, expected single token")
        ids.append(tok[0])
    return ids


def parse_configs(configs: list[str]):
    """
    Convert ['4-16-22', '1-1-29'] → [[4, (16, 22)], [1, (1, 29)]]
    """
    parsed = []
    for cfg in configs:
        try:
            alpha, start, end = map(int, cfg.strip().split("-"))
            parsed.append([alpha, (start, end)])
        except Exception:
            raise ValueError(f"Invalid config format: '{cfg}', should be alpha-start-end (e.g., 4-16-22)")
    return parsed

# ================= Few-shot helpers (prefix-only) =================
INTRO_FMT = "The following are multiple choice questions (with answers) about {subject}."
LABELS = ["A", "B", "C", "D"]

def _fewshot_exemplar(sample: dict, use_E: bool) -> str:
    """
    Construct a single few-shot exemplar block (question + choices + answer).
    """
    labels = LABELS + (["E"] if use_E else [])
    lines = [f"Question: {sample['text']}"]
    choices = sample.get("choices", None)
    if choices is not None:
        assert len(choices) == 4, "choices must be exactly 4 texts (A-D)."
        for i, ch in enumerate(choices):
            lines.append(f"{labels[i]}) {ch}")
        if use_E:
            lines.append("E) I am not sure.")
    else:
        if use_E:
            lines.append("E) I am not sure.")

    ans_idx = sample.get("label", None)
    if ans_idx is None or not (0 <= ans_idx < len(LABELS)):
        raise ValueError("Few-shot exemplar requires a valid 'label' (0..3).")
    lines.append(f"Answer: {LABELS[ans_idx]}")
    return "\n".join(lines)

def build_fewshot_prefix(
    support_pool: List[dict],   # Fixed-order few-shot samples (usually 5)
    k: int,                     # Number of exemplars to include (usually 5)
    use_E: bool,
    subject: Optional[str] = None,
) -> str:
    """
    Build only the few-shot prefix: INTRO + k exemplars (each with Answer).
    Does NOT include the test question; use `build_query_block` to append it separately.
    """
    subj = subject or (support_pool[0].get("task", "task") if support_pool else "task")
    exemplars = support_pool[:k] if k <= len(support_pool) else support_pool

    parts = [INTRO_FMT.format(subject=subj)]
    for s in exemplars:
        parts.append(_fewshot_exemplar(s, use_E=use_E))
        parts.append("")  # blank line separator
    return "\n".join(parts).strip()

def build_query_block(sample: dict, use_E: bool) -> str:
    """
    Build the query block for the test question (without the answer filled in).
    - Adapted for samples where options are already included in `text`.
    """
    lines = [f"Question: {sample['text'].strip()}"]
    if use_E:
        if not sample["text"].strip().endswith("E) I am not sure."):
            lines.append("E) I am not sure.")
    lines.append("Answer:")
    return "\n".join(lines)


if __name__ == "__main__":
    # ===== Data =====
    support_pool = [
        {
            "id": "s1",
            "text": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Rome"],
            "label": 2  # C) Paris
        },
        {
            "id": "s2",
            "text": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "label": 1  # B) 4
        },
        {
            "id": "s3",
            "text": "The largest planet in the Solar System is?",
            "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
            "label": 2  # C) Jupiter
        },
        {
            "id": "s4",
            "text": "Water freezes at what temperature (Celsius)?",
            "choices": ["0", "100", "-10", "50"],
            "label": 0  # A) 0
        },
        {
            "id": "s5",
            "text": "Which element has the chemical symbol O?",
            "choices": ["Oxygen", "Gold", "Osmium", "Oganesson"],
            "label": 0  # A) Oxygen
        },
    ]

    test_sample = {
        "task": "high school computer science",
        "text": "Let x = 1. What is x << 3 in Python 3?\nA) 1\nB) 3\nC) 8\nD) 16\n",
        "label": 2,
      }
      

    fewshot_prefix = build_fewshot_prefix(
        support_pool=support_pool,  
        k=5,
        use_E=False,
        subject="task",
        )
    
    prompt = f"{fewshot_prefix}\n\n{build_query_block(test_sample, use_E=False)}"

    print("===== Few-shot Prompt =====")
    print(prompt)
