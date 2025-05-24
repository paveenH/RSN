#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch answer-generation & accuracy script
Includes gold_text rescue logic
Author: paveenhuang – 2025-05-xx
"""

import os
import json
import re
from tqdm import tqdm

from vicuna import VicundaModel  # Your model wrapper

# ────────────────────── ① Configuration ──────────────────────────────────────────────
TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_medicine",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions"
]

MODEL     = "falcon3"
SIZE      = "7B"
NUM_GPUS  = 1

PATH_MMLU = "/data2/paveen/RolePlaying/src/models/components/mmlu"
SAVE_BASE = "/data2/paveen/RolePlaying/src/models/components/answer_phi"
MODEL_DIR = f"/data2/paveen/RolePlaying/shared/{MODEL}/{SIZE}"

LABELS    = ["A", "B", "C", "D"]  # Option labels

# ────────────────────── ② Generate Character List ────────────────────────────────────────────
def make_characters(task_name: str):
    task_name = task_name.replace("_", " ")
    return [
        f"non-{task_name}",
        f"{task_name}",
    ]

# ────────────────────── ③ Text Cleaning & Extraction ─────────────────────────────────────────
RE_ASSISTANT = re.compile(r"<\|?assistant\|?>", re.I)
RE_PAREN     = re.compile(r"\b([A-E])\s*\)", re.I)
RE_SINGLE    = re.compile(r"\b([A-E])\b")

def sanitize(text: str) -> str:
    """Remove <|assistant|> and other marks, replace newlines, and convert to uppercase"""
    text = RE_ASSISTANT.sub("", text)
    text = text.replace("\n", " ").strip()
    return text.upper()

def extract_choice(raw: str) -> str | None:
    """Attempt to extract A~E; return None if failed"""
    txt = sanitize(raw)
    m = RE_PAREN.search(txt) or RE_SINGLE.search(txt)
    return m.group(1) if m else None

# ────────────────────── ④ Generate & Rescue Logic ─────────────────────────────────────────
def generate_choice(vc, prompt, gold_text: str = None,
                    short_tokens: int = 6, long_tokens: int = 8):
    """
    1. Short generation + extraction; 2. If gold_text is hit → [ADD] correct;
    3. Rescue long generation + same process; 4. Finally fallback to [INV]
    Returns:
      - 'A'..'D'
      - 'E'
      - '[ADD]X ORIGINAL:...' (rescued and gold_text matched)
      - '[INV]...' (other rescue failed)
    """
    # First short generation
    out1 = vc.generate([prompt], max_new_tokens=short_tokens)[0]
    pick = extract_choice(out1)
    if pick:
        return pick
    # E option check
    if gold_text is None:
        # Can recognize E even without gold_text
        if "I AM NOT SURE" in sanitize(out1):
            return "E"
    else:
        # If gold_text appears in the output, directly mark it as correct
        if gold_text.lower() in out1.lower():
            return f"[ADD]{extract_choice(out1) or ''} ORIGINAL:{sanitize(out1)}"

    # Second long generation rescue
    out2 = vc.generate([prompt], max_new_tokens=long_tokens)[0]
    pick2 = extract_choice(out2)
    if pick2:
        # Gold_text takes priority in correct rescue
        if gold_text and gold_text.lower() in out2.lower():
            return f"[ADD]{pick2} ORIGINAL:{sanitize(out2)}"
        return pick2
    # Further E check
    if "I AM NOT SURE" in sanitize(out2):
        return "E"
    # Final fallback
    return f"[INV]{sanitize(out2)}"

# ────────────────────── ⑤ Other Utilities ─────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_full_correct_text(question: str, label_idx: int) -> str | None:
    """
    Extract the full correct option explanation from the question text for gold_text rescue decision
    """
    prefix = f"{LABELS[label_idx]})"
    for line in question.split("\n"):
        if line.strip().upper().startswith(prefix):
            return line.split(prefix, 1)[1].strip().lower()
    return None

def update(acc, ch, tag):
    acc[ch][tag] += 1
    acc[ch]["total"] += 1

# ────────────────────── ⑥ Run Task for Each ─────────────────────────────────────────
def run_task(vc, template: str, task_name: str):
    samples = load_json(os.path.join(PATH_MMLU, f"{task_name}.json"))
    chars   = make_characters(task_name)
    stats   = {c: {"correct":0, "E":0, "invalid":0, "total":0} for c in chars}

    for idx, sample in enumerate(tqdm(samples, desc=task_name)):
        ctx        = sample["text"]
        gold_idx   = sample["label"]
        if not (0 <= gold_idx < len(LABELS)):
            continue
        gold_label = LABELS[gold_idx]
        gold_text  = extract_full_correct_text(ctx, gold_idx)

        for ch in chars:
            prompt = template.format(character=ch, context=ctx)
            ans    = generate_choice(vc, prompt, gold_text)

            # Determine label
            if ans in LABELS:
                tag = "correct" if ans == gold_label else "invalid"
            elif ans == "E":
                tag = "E"
            else:
                tag = "invalid"

            # Statistics & Recording
            update(stats, ch, tag)
            sample[f"answer_{ch.replace(' ','_')}"] = ans

            # Debug output
            if tag == "invalid":
                tqdm.write(f"Sample {idx}, Char '{ch}': invalid '{ans}'")
            if ans.startswith("[ADD]") and tag == "correct":
                tqdm.write(f"[{idx}][{ch}] salvage hit -> Correct")

    # Summarize Accuracy
    summary = {
        ch: {
            **v,
            "accuracy%": round(100 * v["correct"] / v["total"], 2) if v["total"] else 0
        } for ch, v in stats.items()
    }
    return samples, summary

# ────────────────────── ⑦ Main Process ───────────────────────────────────────────
def main():
    print(f"Loading model {MODEL}/{SIZE} …")
    vc       = VicundaModel(model_path=MODEL_DIR, num_gpus=NUM_GPUS)
    template = vc.template
    os.makedirs(SAVE_BASE, exist_ok=True)

    for task in TASKS:
        print(f"\n=== {task} ===")
        data, acc = run_task(vc, template, task)

        # Save Results
        fn = os.path.join(SAVE_BASE, f"{task}_{SIZE}_answers.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({"data": data, "accuracy": acc}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {fn}")

        # Print Summary
        for ch, stats in acc.items():
            print(f"{ch:>22}: {stats['accuracy%']}% "
                  f"(✔{stats['correct']}/{stats['total']} "
                  f"E{stats['E']} ✗{stats['invalid']})")

    print("\n✅ All tasks finished!")

if __name__ == "__main__":
    main()