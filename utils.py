#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:29:48 2025

@author: paveenhuang
"""

import re
import json
import hashlib
import random
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

# ================= Few-shot helpers (MMLU-style) =================

INTRO_FMT = "The following are multiple choice questions (with answers) about {subject}."

def stable_seed(*parts, global_seed: int = 0) -> int:
    """
    Deterministic seed generator from stringable parts + global_seed.
    """
    h = hashlib.sha256(("||".join(map(str, parts)) + f"||{global_seed}").encode()).hexdigest()
    return int(h[:8], 16)  # first 32 bits

def _fewshot_exemplar(sample: dict, use_E: bool) -> str:
    labels = ["A","B","C","D","E"] if use_E else ["A","B","C","D"]
    lines = []
    lines.append(f"Question: {sample['text']}")
    choices = sample.get("choices", None)
    if choices is not None:
        for i, ch in enumerate(choices):
            lines.append(f"{labels[i]}) {ch}")
        if use_E and len(choices) == 4:
            lines.append("E) I am not sure.")
    else:
        if use_E:
            lines.append("E) I am not sure.")
    # answer label index is expected as 'label' (0-based)
    ans_idx = sample.get("label", None)
    if ans_idx is None:
        raise ValueError("Few-shot exemplar requires 'label' in sample.")
    lines.append(f"Answer: {labels[ans_idx]}")
    return "\n".join(lines)

def _fewshot_query_block(sample: dict, use_E: bool) -> str:
    labels = ["A","B","C","D","E"] if use_E else ["A","B","C","D"]
    lines = []
    lines.append(f"Question: {sample['text']}")
    choices = sample.get("choices", None)
    if choices is not None:
        for i, ch in enumerate(choices):
            lines.append(f"{labels[i]}) {ch}")
        if use_E and len(choices) == 4:
            lines.append("E) I am not sure.")
    else:
        if use_E:
            lines.append("E) I am not sure.")
    lines.append("Answer:")
    return "\n".join(lines)

def build_fewshot_prompt(
    test_sample: dict,
    support_pool: List[dict],
    k: int,
    use_E: bool,
    tokenizer,
    max_tokens: int = 8192,
    global_seed: int = 0,
    subject: Optional[str] = None,
) -> str:
    """
    Construct harness-style few-shot prompt for a given test sample.
    - support_pool: list of candidate exemplars (should not include test_sample)
    - k: number of exemplars
    - subject: optional display name in the INTRO line (falls back to 'task')
    """
    subj = subject or "task"
    # Deterministic shuffle
    rnd = random.Random(stable_seed(subj, test_sample.get("id", ""), global_seed))
    pool = [s for s in support_pool if s is not test_sample]
    rnd.shuffle(pool)
    exemplars = pool[:k] if k <= len(pool) else pool

    parts = [INTRO_FMT.format(subject=subj)]
    for s in exemplars:
        parts.append(_fewshot_exemplar(s, use_E=use_E))
        parts.append("")  # blank line

    parts.append(_fewshot_query_block(test_sample, use_E=use_E))
    prompt = "\n".join(parts).strip()

    # Length control (rough cut: drop earliest exemplars until within limit)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(ids) > max_tokens:
        for kk in range(len(exemplars) - 1, -1, -1):
            parts = [INTRO_FMT.format(subject=subj)]
            for s in exemplars[:kk]:
                parts.append(_fewshot_exemplar(s, use_E=use_E))
                parts.append("")
            parts.append(_fewshot_query_block(test_sample, use_E=use_E))
            prompt2 = "\n".join(parts).strip()
            if len(tokenizer(prompt2, add_special_tokens=False).input_ids) <= max_tokens:
                prompt = prompt2
                break

    return prompt


if __name__ == "__main__":
    from transformers import AutoTokenizer

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
        "id": "test1",
        "text": "Who wrote 'Pride and Prejudice'?",
        "choices": ["Jane Austen", "Emily Brontë", "Charles Dickens", "Mark Twain"],
        "label": 0  # A) Jane Austen
    }

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    prompt = build_fewshot_prompt(
        test_sample=test_sample,
        support_pool=support_pool,
        k=3,                  
        use_E=False,          
        tokenizer=tokenizer,
        global_seed=42,
        subject="General Knowledge"
    )

    print("===== Few-shot Prompt =====")
    print(prompt)
