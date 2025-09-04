#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:16:47 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert FACTOR / FACTors style data into MMLU-Pro-like multiple-choice JSON.

Output per item:
- task        : str  (e.g., "FACTOR (binary)" / "FACTOR (error-type)" / "FACTors (verdict)")
- category    : str  (domain/category if available; else use a default)
- text        : str  "Question...\nA) ...\nB) ...\n..."
- label       : int  (0-based correct option index)
- num_options : int  (# of options)

USAGE:
- Put your raw FACTOR / FACTors data as JSON or JSONL under input files.
- Configure the paths and mappings in the "CONFIGURATION" block below.
- Run this script. It will write the converted list to the output JSON path.

NOTE:
- This script avoids external dependencies beyond the standard library.
- It is schema-flexible via small mapping lambdas you can adapt in the CONFIGURATION.
"""

import os
import json
from typing import Any, Dict, List

# =========================
# CONFIGURATION (edit here)
# =========================

# Choose which converter(s) to run
RUN_FACTOR_BINARY      = True   # FACTOR → binary: Correct vs. Factual error
RUN_FACTOR_ERRORTYPE   = False  # FACTOR → multi-class: error-type classification
RUN_FACTORS_VERDICT    = True   # FACTors → multi-class: True/False/Partially true/Unverifiable

# Input paths (set to your actual files)
FACTOR_INPUT_PATH      = "/path/to/factor.jsonl"      # or .json
FACTORS_INPUT_PATH     = "/path/to/factors.jsonl"     # or .json

# Output paths
OUT_FACTOR_BINARY      = "/path/to/output/factor_binary.json"
OUT_FACTOR_ERRORTYPE   = "/path/to/output/factor_errortype.json"
OUT_FACTORS_VERDICT    = "/path/to/output/factors_verdict.json"

# Default categories when missing
DEFAULT_CATEGORY_FACTOR   = "factuality"
DEFAULT_CATEGORY_FACTORS  = "fact-checking"

# Letters for options
LETTER10 = ["A","B","C","D","E","F","G","H","I","J"]


# =========================================
# Helpers: file loading / saving / utilities
# =========================================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a .json (list or dict with 'data') or .jsonl file into a list of dicts.
    """
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    # Fall back: wrap single dict
    return [obj]

def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_mc_text(question: str, options: List[str], letters: List[str]) -> str:
    text = (question or "").strip()
    K = min(len(options), len(letters))
    for i in range(K):
        text += f"\n{letters[i]}) {options[i]}"
    return text + "\n"


# =========================================
# FACTOR → Multiple-choice converters
# =========================================

# ---- FACTOR (binary) mapping ----
# Adapt these lambdas to your actual schema.
# We assume each sample has fields like:
#   - "prompt" or "question" or "input"
#   - "model_output" or "prediction" or "text"
#   - "is_factual" (bool) or "label" in {"correct","incorrect"} / {0,1}
FACTOR_BINARY_FIELD_MAP = {
    "get_question": lambda row: (
        # You can enrich the question as you like:
        # put the prompt and model output together to ask a binary question.
        f"Does the following statement contain a factual error?\n\n"
        f"Statement: {(row.get('model_output') or row.get('output') or row.get('prediction') or row.get('text') or '').strip()}"
    ),
    "get_category": lambda row: row.get("domain") or row.get("source") or DEFAULT_CATEGORY_FACTOR,
    "get_is_error": lambda row: (
        # Return True if the statement is factually wrong.
        # Prioritize explicit fields; customize to your data schema.
        (False if row.get("is_factual") is True else True) if "is_factual" in row
        else (True if str(row.get("label","")).lower() in {"error","incorrect","false","f"} else None)
    ),
}

FACTOR_BINARY_OPTIONS = [
    "No, it is factually correct",
    "Yes, it contains a factual error",
]
# Label: 0 => correct, 1 => error


# ---- FACTOR (error-type) mapping ----
# Here we classify into multiple error categories, e.g.:
#   "correct", "wrong_entity", "wrong_number", "wrong_relation", ...
# You must align this to your dataset's actual taxonomy.
FACTOR_ERRORTYPE_FIELD_MAP = {
    "get_question": lambda row: (
        f"What best describes the factual status of the statement below?\n\n"
        f"Statement: {(row.get('model_output') or row.get('output') or row.get('prediction') or row.get('text') or '').strip()}"
    ),
    "get_category": lambda row: row.get("domain") or row.get("source") or DEFAULT_CATEGORY_FACTOR,
    "get_errortype": lambda row: str(row.get("error_type") or row.get("label") or "correct").strip().lower(),
}

# Define your canonical option set (order matters)
FACTOR_ERRORTYPE_OPTIONS = [
    "Correct",
    "Wrong entity",
    "Wrong number",
    "Wrong relation",
    "Outdated information",
    "Unverifiable",
]
# Map dataset-specific tags → above options
FACTOR_ERRORTYPE_NORMALIZE = {
    "correct": "Correct",
    "ok": "Correct",
    "no_error": "Correct",
    "factual": "Correct",
    "wrong_entity": "Wrong entity",
    "entity": "Wrong entity",
    "wrong_number": "Wrong number",
    "number": "Wrong number",
    "wrong_relation": "Wrong relation",
    "relation": "Wrong relation",
    "outdated": "Outdated information",
    "stale": "Outdated information",
    "unverifiable": "Unverifiable",
    "not_verifiable": "Unverifiable",
}


def convert_factor_binary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data_out: List[Dict[str, Any]] = []
    for r in rows:
        question = FACTOR_BINARY_FIELD_MAP["get_question"](r)
        category = FACTOR_BINARY_FIELD_MAP["get_category"](r)
        is_error = FACTOR_BINARY_FIELD_MAP["get_is_error"](r)

        # If we cannot infer, skip or default. Here we skip.
        if is_error is None:
            # You can log or set a default: is_error = True/False
            continue

        options = FACTOR_BINARY_OPTIONS
        # 0 => "No, it is factually correct"; 1 => "Yes, ... error"
        label = 1 if is_error else 0
        text = format_mc_text(question, options, LETTER10)
        data_out.append({
            "task": "FACTOR (binary)",
            "category": category,
            "text": text,
            "label": int(label),
            "num_options": len(options),
        })
    return data_out


def convert_factor_errortype(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data_out: List[Dict[str, Any]] = []
    for r in rows:
        question = FACTOR_ERRORTYPE_FIELD_MAP["get_question"](r)
        category = FACTOR_ERRORTYPE_FIELD_MAP["get_category"](r)
        et_raw = FACTOR_ERRORTYPE_FIELD_MAP["get_errortype"](r)
        et_norm = FACTOR_ERRORTYPE_NORMALIZE.get(et_raw, None)
        if et_norm is None:
            # Unknown tag → fallback to "Unverifiable" or "Correct"
            et_norm = "Unverifiable" if et_raw not in {"correct", "ok", "no_error", "factual"} else "Correct"

        options = FACTOR_ERRORTYPE_OPTIONS
        try:
            label = options.index(et_norm)
        except ValueError:
            # If still not found, place in "Unverifiable"
            label = options.index("Unverifiable")

        text = format_mc_text(question, options, LETTER10)
        data_out.append({
            "task": "FACTOR (error-type)",
            "category": category,
            "text": text,
            "label": int(label),
            "num_options": len(options),
        })
    return data_out


# =========================================
# FACTors → Multiple-choice converter
# =========================================

# Typical FACTors row may contain:
#   - "claim" / "title"
#   - "verdict" / "rating" / "label"  (e.g., true/false/partly true/unverifiable)
#   - optional "source", "domain", "organization"
FACTORS_FIELD_MAP = {
    "get_question": lambda row: (
        f"According to fact-checkers, what is the verdict for the following claim?\n\n"
        f"Claim: {(row.get('claim') or row.get('title') or row.get('text') or '').strip()}"
    ),
    "get_category": lambda row: row.get("domain") or row.get("organization") or DEFAULT_CATEGORY_FACTORS,
    "get_verdict":  lambda row: str(row.get("verdict") or row.get("rating") or row.get("label") or "").strip().lower(),
}

FACTORS_OPTIONS = ["True", "False", "Partially true", "Unverifiable"]

FACTORS_NORMALIZE = {
    "true": "True",
    "mostly true": "True",
    "correct": "True",
    "false": "False",
    "pants on fire": "False",
    "fake": "False",
    "partly true": "Partially true",
    "half true": "Partially true",
    "mixture": "Partially true",
    "unverifiable": "Unverifiable",
    "insufficient evidence": "Unverifiable",
    "uncertain": "Unverifiable",
}

def convert_factors_verdict(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data_out: List[Dict[str, Any]] = []
    for r in rows:
        question = FACTORS_FIELD_MAP["get_question"](r)
        category = FACTORS_FIELD_MAP["get_category"](r)
        v_raw = FACTORS_FIELD_MAP["get_verdict"](r)
        v_norm = FACTORS_NORMALIZE.get(v_raw, None)
        if v_norm is None:
            # default fallback if unknown
            v_norm = "Unverifiable"

        options = FACTORS_OPTIONS
        try:
            label = options.index(v_norm)
        except ValueError:
            label = options.index("Unverifiable")

        text = format_mc_text(question, options, LETTER10)
        data_out.append({
            "task": "FACTors (verdict)",
            "category": category,
            "text": text,
            "label": int(label),
            "num_options": len(options),
        })
    return data_out


# =========================
# Main execution
# =========================

if __name__ == "__main__":
    if RUN_FACTOR_BINARY:
        factor_rows = load_json_or_jsonl(FACTOR_INPUT_PATH)
        out = convert_factor_binary(factor_rows)
        save_json(OUT_FACTOR_BINARY, out)
        print(f"[OK] FACTOR (binary) → {OUT_FACTOR_BINARY} | {len(out)} items")

    if RUN_FACTOR_ERRORTYPE:
        factor_rows = load_json_or_jsonl(FACTOR_INPUT_PATH)
        out = convert_factor_errortype(factor_rows)
        save_json(OUT_FACTOR_ERRORTYPE, out)
        print(f"[OK] FACTOR (error-type) → {OUT_FACTOR_ERRORTYPE} | {len(out)} items")

    if RUN_FACTORS_VERDICT:
        factors_rows = load_json_or_jsonl(FACTORS_INPUT_PATH)
        out = convert_factors_verdict(factors_rows)
        save_json(OUT_FACTORS_VERDICT, out)
        print(f"[OK] FACTors (verdict) → {OUT_FACTORS_VERDICT} | {len(out)} items")