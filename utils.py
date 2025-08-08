#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:29:48 2025

@author: paveenhuang
"""

import re
import json


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
            # f"non {task_name} expert",
            # f"{task_name} expert",
            # # f"not an expert in {task_name}",
            f"{task_name} student",
            "person",
            # "norole"
        ]
    else:
        return


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
