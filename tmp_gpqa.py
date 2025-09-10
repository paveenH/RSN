#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset
import json
import random

# ---------------- Config ----------------
DATASET_NAME = "Idavidrein/gpqa"
CONFIG_NAME  = "gpqa_main"     # one of: gpqa_main, gpqa_diamond, gpqa_extended, gpqa_experts
SPLIT        = "train"         
N_SAMPLES    = 5               # how many to preview
SEED         = 42
# ----------------------------------------


LETTER24 = [chr(ord("A") + i) for i in range(24)]

def pick_first_nonempty(row, keys):
    """Return first non-empty string among candidate keys."""
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    return ""

def list_from_row(row, keys):
    """Collect a list from row[k] when present and non-empty strings."""
    vals = []
    for k in keys:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, (list, tuple)):
                vals.extend([str(x).strip() for x in v if str(x).strip()])
            else:
                s = str(v).strip()
                if s:
                    vals.append(s)
    return vals

def normalize_options(options):
    """Remove empties / strip and truncate to <=24."""
    opts = [str(x).strip() for x in options if str(x).strip()]
    return opts[:len(LETTER24)]

def try_extract_question(row):
    # try common fields for question text
    return pick_first_nonempty(
        row,
        ["question", "Question", "prompt", "stem", "text", "input"]
    )

def try_extract_options_and_correct(row):
    """
    Try multiple schema patterns to get (options:list, correct_value:str or index:int).
    Return (options, correct_value_or_index, is_index_bool).
    """
    # Pattern A: ready-made options list + answer index
    if "options" in row and isinstance(row["options"], (list, tuple)):
        opts = normalize_options(row["options"])
        # answer could be index, letter, or exact string
        ans = row.get("answer", row.get("Answer", None))
        if ans is not None:
            # index?
            try:
                idx = int(ans)
                if 0 <= idx < len(opts):
                    return opts, idx, True
            except Exception:
                pass
            # letter?
            s = str(ans).strip()
            if len(s) == 1 and "A" <= s.upper() <= "Z":
                idx = ord(s.upper()) - ord("A")
                if 0 <= idx < len(opts):
                    return opts, idx, True
            # exact string match?
            s = str(ans).strip()
            if s in opts:
                return opts, s, False
        # no explicit answer: fall back to other patterns

    # Pattern B: choices/answers/answer_choices
    for key in ["choices", "answers", "answer_choices"]:
        if key in row and isinstance(row[key], (list, tuple)):
            opts = normalize_options(row[key])
            ans = row.get("answer", row.get("Answer", None))
            if ans is not None:
                # try index/letter/string as above
                try:
                    idx = int(ans)
                    if 0 <= idx < len(opts):
                        return opts, idx, True
                except Exception:
                    pass
                s = str(ans).strip()
                if len(s) == 1 and "A" <= s.upper() <= "Z":
                    idx = ord(s.upper()) - ord("A")
                    if 0 <= idx < len(opts):
                        return opts, idx, True
                if s in opts:
                    return opts, s, False

    # Pattern C: correct + incorrect list
    correct_val = pick_first_nonempty(row, ["correct_answer", "Correct Answer", "answer_text", "gold", "label_text"])
    incorrect_vals = list_from_row(row, ["incorrect_answers", "Incorrect Answers"])
    if not incorrect_vals:
        # sometimes incorrects are split across columns, e.g., A/B/C/D style with one of them being correct
        abcd = list_from_row(row, ["A", "B", "C", "D", "E"])
        # If we also have an answer letter, we can form options in AB.. order.
        if abcd:
            # If they also provide "answer" as letter/index, we can use it directly:
            ans = row.get("answer", row.get("Answer", None))
            if ans is not None:
                try:
                    idx = int(ans)
                    if 0 <= idx < len(abcd):
                        return normalize_options(abcd), idx, True
                except Exception:
                    pass
                s = str(ans).strip()
                if len(s) == 1 and "A" <= s.upper() <= "Z":
                    idx = ord(s.upper()) - ord("A")
                    if 0 <= idx < len(abcd):
                        return normalize_options(abcd), idx, True
            # else: we cannot identify which is correct—return empty to signal failure
            return [], None, False

    if correct_val and incorrect_vals:
        opts = normalize_options([correct_val] + incorrect_vals)
        return opts, correct_val, False

    # Pattern D: one field with JSON-like options dict (rare fallback)
    # If needed, you can add a json.loads attempt here.

    # If nothing worked:
    return [], None, False

def build_text(question, options):
    text = question.strip()
    for i, opt in enumerate(options):
        text += f"\n{LETTER24[i]}) {opt}"
    return text + "\n"

def main():
    rnd = random.Random(SEED)

    # Load dataset
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
    print(f"Loaded GPQA ({CONFIG_NAME}/{SPLIT}) with {len(ds)} samples.\n")

    # Inspect schema quickly (helps you see real column names)
    print("Columns:", ds.column_names)
    print("First row (raw):", ds[0], "\n")

    samples = []
    for i in range(min(N_SAMPLES, len(ds))):
        row = ds[i]

        # 1) question
        q = try_extract_question(row)

        # 2) options + correct
        options, ans_val, ans_is_index = try_extract_options_and_correct(row)

        # if we still failed, just produce a placeholder to debug
        if not q or not options or ans_val is None:
            samples.append({
                "task": f"GPQA ({CONFIG_NAME})",
                "category": "science",
                "text": "",     # leave empty so you notice it
                "label": -1,
                "num_options": 0,
                "_debug_row": row,  # include for your debugging
            })
            continue

        # 3) find correct index BEFORE shuffling (if ans is value)
        if ans_is_index:
            correct_idx = int(ans_val)
        else:
            try:
                correct_idx = options.index(str(ans_val).strip())
            except ValueError:
                # cannot find exact match → mark for debugging
                samples.append({
                    "task": f"GPQA ({CONFIG_NAME})",
                    "category": "science",
                    "text": "",
                    "label": -1,
                    "num_options": 0,
                    "_debug_note": "Correct value not in options",
                    "_debug_row": row,
                })
                continue

        # 4) shuffle options and map label
        perm = list(range(len(options)))
        rnd.shuffle(perm)
        options_shuf = [options[j] for j in perm]
        new_label = perm.index(correct_idx)

        # 5) build MMLU-Pro-like text
        text = build_text(q, options_shuf)

        samples.append({
            "task": f"GPQA ({CONFIG_NAME})",
            "category": "science",
            "text": text,
            "label": new_label,
            "num_options": len(options_shuf),
        })

    print(json.dumps(samples, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()