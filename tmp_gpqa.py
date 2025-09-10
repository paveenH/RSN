#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset
import json, random

# ======== Config ========
DATASET_NAME = "Idavidrein/gpqa"
CONFIG_NAME  = "gpqa_main"        # gpqa_main | gpqa_diamond | gpqa_extended | gpqa_experts
SPLIT        = "train"            # GPQA 通常只有 train split
N_SAMPLES    = 5
SEED         = 42
# ========================

LETTER = [chr(ord('A')+i) for i in range(24)]
rnd = random.Random(SEED)

def first_nonempty(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            s = str(row[k]).strip()
            if s:
                return s
    return ""

def list_from(row, keys):
    out = []
    for k in keys:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, (list, tuple)):
                out += [str(x).strip() for x in v if str(x).strip()]
            else:
                s = str(v).strip()
                if s:
                    out.append(s)
    return out

def try_extract_qa(row):
    """Return (question:str, options:list[str], correct_index:int) or (None,[],None)"""
    # 1) question candidates
    q = first_nonempty(row, ["Question", "question", "prompt", "stem", "text"])
    # 2) ready-made options + answer
    if "options" in row and isinstance(row["options"], (list, tuple)):
        opts = [str(x).strip() for x in row["options"] if str(x).strip()]
        if opts:
            a = row.get("answer", row.get("Answer", None))
            if a is not None:
                # index?
                try:
                    idx = int(a)
                    if 0 <= idx < len(opts):
                        return q, opts, idx
                except Exception:
                    pass
                # letter?
                s = str(a).strip()
                if len(s) == 1 and "A" <= s.upper() <= "Z":
                    idx = ord(s.upper())-ord("A")
                    if 0 <= idx < len(opts):
                        return q, opts, idx
                # exact string?
                if str(a).strip() in opts:
                    return q, opts, opts.index(str(a).strip())

    # 3) correct + incorrects pattern (GPQA 常见)
    correct = first_nonempty(row, ["Correct Answer", "correct_answer", "Answer", "answer_text"])
    incorrects = list_from(row, ["incorrect_answers", "Incorrect Answers"])
    if not incorrects:
        inc_keys = [k for k in row.keys() if k.lower().startswith("incorrect answer")]
        inc_keys = sorted(inc_keys) 
        incorrects = list_from(row, inc_keys)

    if correct and incorrects:
        opts = [correct] + incorrects
        return q, opts, 0

    abcd = list_from(row, ["A", "B", "C", "D", "E"])
    if abcd:
        a = row.get("answer", row.get("Answer", None))
        if a is not None:
            try:
                idx = int(a)
                if 0 <= idx < len(abcd):
                    return q, abcd, idx
            except Exception:
                pass
            s = str(a).strip()
            if len(s) == 1 and "A" <= s.upper() <= "Z":
                idx = ord(s.upper())-ord("A")
                if 0 <= idx < len(abcd):
                    return q, abcd, idx

    return None, [], None

def build_text(q, options):
    text = (q or "").strip()
    for i, opt in enumerate(options):
        text += f"\n{LETTER[i]}) {opt}"
    return text + "\n"

def main():
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT)
    print(f"Loaded GPQA ({CONFIG_NAME}/{SPLIT}) with {len(ds)} samples.")
    useful_cols = [c for c in ds.column_names
                   if c.lower().startswith(("question","correct","incorrect","answer","options","a","b","c","d","e"))]
    print("Useful-ish columns:", useful_cols)

    out = []
    misses = 0
    for i in range(min(N_SAMPLES, len(ds))):
        row = ds[i]
        q, opts, gold = try_extract_qa(row)
        if not q or not opts or gold is None:
            misses += 1
            out.append({
                "task": f"GPQA ({CONFIG_NAME})",
                "category": "science",
                "text": "",
                "label": -1,
                "num_options": 0,
                "_debug_keys": list(row.keys())[:20]  # 只放前 20 个键，避免刷屏
            })
            continue

        perm = list(range(len(opts)))
        rnd.shuffle(perm)
        opts_shuf = [opts[j] for j in perm]
        label = perm.index(gold)

        item = {
            "task": f"GPQA ({CONFIG_NAME})",
            "category": "science",
            "text": build_text(q, opts_shuf),
            "label": label,
            "num_options": len(opts_shuf),
        }
        out.append(item)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if misses:
        print(f"\n[Note] {misses} / {min(N_SAMPLES, len(ds))} samples were missing core fields and printed with _debug_keys.")

if __name__ == "__main__":
    main()