#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified merger for AGIEval MCQ → MMLU-Pro-like JSON.

Input:  /data2/paveen/RolePlaying/components/agieval/*.jsonl
Output: /data2/paveen/RolePlaying/components/agieval_mcq.json

Each item:
{
  "task": "AGIEval-<filename>",
  "text": "<passage?>\\n\\n<question>\\nA) ...\\nB) ...\\n...",
  "label": <0-based index>,
  "label_letter": "A/B/...",
  "num_options": K
}
"""

import os, json, glob

INPUT_DIR = "/data2/paveen/RolePlaying/components/agieval"
OUT_PATH  = "/data2/paveen/RolePlaying/components/agieval_mcq/agieval_mcq.json"

LETTER = [chr(ord("A")+i) for i in range(26)]
SKIP_BASENAMES = {"gaokao-mathcloze", "math"}  # cloze / not MCQ

def label_to_index(label, nopt):
    if label is None: return None
    s = str(label).strip().upper()
    for ch in s:
        if "A" <= ch <= "Z":
            idx = ord(ch) - ord("A")
            return idx if 0 <= idx < nopt else None
    return None

def build_text(passage, question, options):
    lines = []
    if passage:
        lines.append(str(passage).strip())
        lines.append("")
    lines.append(question.strip())
    for i, opt in enumerate(options):
        lines.append(f"{LETTER[i]}) {opt.strip()}")
    return "\n".join(lines) + "\n"

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out

def main():
    merged = []
    for fp in sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))):
        base = os.path.splitext(os.path.basename(fp))[0]
        if base in SKIP_BASENAMES:
            print(f"[SKIP] {base} (cloze / non-MCQ)")
            continue
        records = load_jsonl(fp)
    
        for rec in records:
            if "options" not in rec or "label" not in rec: 
                continue
            options = [str(x).strip() for x in rec["options"]]
            gold = label_to_index(rec["label"], len(options))
            if gold is None: 
                continue
            text = build_text(rec.get("passage"), rec["question"], options)
            merged.append({
                "task": f"AGIEval-{base}",
                "text": text,
                "label": gold,
                "label_letter": str(rec["label"]).strip(),
                "num_options": len(options),
            })
        print(f"[INFO] {base}: {len(records)} items")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved merged AGIEval MCQ → {OUT_PATH}")
    print(f"Total items: {len(merged)}")

if __name__ == "__main__":
    main()