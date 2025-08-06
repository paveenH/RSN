#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch answer-generation & accuracy script with diffusion support.
Loops over TASKS and roles, applies templates, generates answers, and computes accuracy.
Author: paveenhuang (refactored)
"""

import os
import json
import re
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates


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
            f"non {task_name}",
            f"{task_name}",
        ]
    else:
        return


def generate_answer(vc, prompt: str, diffusion_mode: str, short: int, step: int):
    if diffusion_mode == "dream":
        out = vc.generate_diffusion_dream(
            [prompt],
            max_new_tokens=short,
            steps=step,
            top_p=1,
            temperature=0,
        )[0]
    elif diffusion_mode == "llada":
        out = vc.generate_diffusion_llada(
            [prompt],
            max_new_tokens=short,
            steps=step,
            block_len=short,
        )[0]
    else:
        out = vc.generate([prompt], max_new_tokens=short)[0]
    return cleaning(out)


def handle_invalid_answer(vc, prompt: str, true_text: str, true_label: str,
                          diffusion_mode: str, short: int, long: int, step: int):
    # retry with longer generation
    if diffusion_mode == "dream":
        out_long = vc.generate_diffusion_dream(
            [prompt],
            max_new_tokens=long,
            steps=step,
            top_p=1,
            temperature=0,
        )[0].strip()
    elif diffusion_mode == "llada":
        out_long = vc.generate_diffusion_llada(
            [prompt],
            max_new_tokens=long,
            steps=step,
            block_len=long,
        )[0].strip()
    else:
        out_long = vc.generate([prompt], max_new_tokens=long)[0].strip()

    out_long = out_long.replace("<|assistant|>", "").replace("\u200b", "").strip().upper()
    extracted = cleaning(out_long)
    
    if extracted in LABEL_MAPPING:
        if extracted == true_label:
            return "[Add]" + extracted + out_long, True, False
        else:
            return extracted + out_long, False, False

    if extracted == "E":
        return "[Add]" + out_long, False, True

    if true_text and true_text.lower() in out_long.lower():
        return "[Add]" + out_long + "contains" + true_text, True, False

    if "i am not sure" in out_long.lower():
        return "[Add]" + out_long, False, True

    return out_long, False, False



def update(acc, char, status):
    acc[char][status] += 1


def run_task(vc: VicundaModel,
             templates: dict,
             task: str,
             short: int,
             long: int,
             step: int,
             diffusion_mode: str):
    """Run one MMLU task, generate answers for each role, return updated data + accuracy."""
    LABELS = templates["labels"]
    default_t   = templates["default"]
    neutral_t   = templates["neutral"]
    neg_t       = templates["neg"]
    vanilla_t   = templates["vanilla"]

    data = load_json(os.path.join(MMLU_DIR, f"{task}.json"))
    chars = make_characters(task, args.type)
    print("characters:", chars)

    # stats initialization
    acc = {c: {"correct": 0, "E": 0, "invalid": 0, "total": 0} for c in chars}

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data, desc=task)):
            ctx = sample["text"]
            true_idx = sample["label"]
            if not (0 <= true_idx < len(LABELS)):
                continue
            true_label = LABELS[true_idx]
            # extract full answer text if needed for rescue
            true_text = None
            prefix = f"{true_label})"
            for line in ctx.split("\n"):
                if line.strip().upper().startswith(prefix):
                    true_text = line.strip()[len(prefix):].strip().lower()
                    break

            for ch in chars:
                # choose prompt template
                if ch == "norole":
                    prompt = neutral_t.format(context=ctx)
                elif ch == "vanilla":
                    prompt = vanilla_t.format(context=ctx)
                elif "not" in ch:
                    prompt = neg_t.format(character=ch, context=ctx)
                else:
                    prompt = default_t.format(character=ch, context=ctx)

                # generate and clean
                ans = generate_answer(vc, prompt, diffusion_mode, short, step)
                # rescue invalid
                if ans not in LABELS and ans != "E":
                    ans, is_corr, is_E = handle_invalid_answer(
                        vc, prompt, true_text, true_label,
                        diffusion_mode, short, long, step
                    )
                    if is_corr:
                        status = "correct"
                        tqdm.write(f"[{idx}][{ch}] '{ans}' -> Correct")
                    elif is_E:
                        status = "E"
                        tqdm.write(f"[{idx}][{ch}] '{ans}' -> E")
                    else:
                        status = "invalid"
                        tqdm.write(f"[{idx}][{ch}] '{ans}' -> Invalid")
                else:
                    status = "correct" if ans == true_label else ("E" if ans == "E" else "invalid")

                acc[ch]["total"] += 1
                update(acc, ch, status)
                sample[f"answer_{ch.replace(' ', '_')}"] = ans

    # build summary
    summary = {}
    for ch, stats in acc.items():
        total = stats["total"]
        correct = stats["correct"]
        pct = (correct / total * 100) if total else 0.0
        summary[ch] = {
            "correct": correct,
            "E_count": stats["E"],
            "invalid": stats["invalid"],
            "total": total,
            "accuracy_percentage": round(pct, 2),
        }
    return data, summary


def main():
    print(f"Loading model {MODEL}/{SIZE}…")
    vc = VicundaModel(model_path=MODEL_DIR, diffusion_mode=DIFFUSION)
    templates = select_templates(args.use_E)
    vc.model.eval()

    save_dir = SAVE_BASE / MODEL
    save_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        print(f"\n=== {task} ===")
        data, acc = run_task(
            vc,
            templates,
            task,
            args.short,
            args.long,
            STEP,
            DIFFUSION
        )

        out = save_dir / f"{task}_{SIZE}_answers.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"data": data, "accuracy": acc}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out}")

        for ch, r in acc.items():
            print(
                f"{ch:>18}: {r['accuracy_percentage']}% "
                f"(correct {r['correct']}/{r['total']}, "
                f"E {r['E_count']}, invalid {r['invalid']})"
            )

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MMLU role-based answer gen with optional hidden-state saving"
    )
    parser.add_argument("--model",   "-m", required=True, help="Model name for folder naming")
    parser.add_argument("--size",    "-s", required=True, help="Model size, e.g., `8B`")
    parser.add_argument("--type",           required=True, help="Role type identifier")
    parser.add_argument("--model_dir",      required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--ans_file",       required=True, help="Output base folder name")
    parser.add_argument("--use_E",   action="store_true", help="Use five-choice template (A–E)")
    parser.add_argument("--save",    action="store_true", help="Save hidden states (not used here)")
    parser.add_argument("--short",   type=int, default=2,  help="Max tokens for short gen")
    parser.add_argument("--long",    type=int, default=12, help="Max tokens for long rescue")
    args = parser.parse_args()

    # experiment constants
    STEP      = 16
    DIFFUSION = None  # or "dream"/"llada"
    LABEL_MAPPING = ["A", "B", "C", "D"]

    MODEL     = args.model
    SIZE      = args.size
    TYPE      = args.type
    MODEL_DIR = args.model_dir
    SAVE_BASE = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}")
    MMLU_DIR  = Path("/data2/paveen/RolePlaying/components/mmlu")

    main()