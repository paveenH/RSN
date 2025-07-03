#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel with neuron editing across multiple tasks.
Loads the model(s) once and processes all combinations of tasks, top-k,
α values, and layer ranges in a single run.
@author: paveenhuang
"""

import os
import json
import numpy as np
from tqdm import tqdm
import torch

from llms import VicundaModel
import get_answer_alltasks as ga


LABEL_MAPPING = ["A", "B", "C", "D"]

# === Configuration ===
TASKS = ga.TASKS

MODEL = "phi4"
HiddenModel = "phi4"
SIZE = "4B"
TYPE = "non"
# 
# MODEL_DIR = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_DIR = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_DIR = "NousResearch/Hermes-3-Llama-3.2-3B"
# MODEL_DIR = "openchat/openchat_3.5"
# MODEL_DIR =  "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_DIR = "HuggingFaceH4/zephyr-7b-beta"
# MODEL_DIR = "Qwen/Qwen2.5-3B-Instruct"
# MODEL_DIR = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = "microsoft/Phi-4-mini-instruct"

print(MODEL_DIR)

TOP = 15
# ALPHAS_START_END_PAIRS = [[4, (21,30)], [1, (1,37)]]
ALPHAS_START_END_PAIRS = [[4, (7,15)], [3, (7,15)], [1, (1,33)]]

SHORT = 2
LONG = 12

DIFFUSION = None  # dream/ llada/ None
STEP = 16

MAT_DIR = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HiddenModel}_{TYPE}"
print("import hidden states difference from ", MAT_DIR)
MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
SAVE_DIR = f"/data2/paveen/RolePlaying/components/answer_modified_{TYPE}"


# === Helper functions ===
def build_char_diff(alpha: int, start: int, end: int):
    try:
        diff_char = np.load(os.path.join(MAT_DIR, f"diff_mean_{SIZE}.npy"))
        diff_none = np.load(os.path.join(MAT_DIR, f"none_diff_mean_{SIZE}.npy"))
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing diff matrix: {e}")

    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (layers, hidden)

    for layer in range(diff.shape[0]):
        if start <= layer < end:
            layer_vec = diff[layer]
            idxs = np.argsort(np.abs(layer_vec))[-TOP:]
            mask = np.zeros_like(layer_vec, dtype=bool)
            mask[idxs] = True
            diff[layer] = layer_vec * mask
        else:
            diff[layer] = 0

    return diff[1:] * alpha


def generate_answer_diff(vc, prompt: str, diff_mtx):
    out = vc.regenerate([prompt], diff_matrices=diff_mtx, max_new_tokens=SHORT)[0]
    return ga.cleaning(out)


def handle_invalid_answer_diff(vc, prompt: str, true_text: str, true_label: str, diff_mtx):
    out = vc.regenerate([prompt], diff_matrices=diff_mtx, max_new_tokens=LONG)[0]
    out = out.replace("<|assistant|>", "").replace("\u200b", "").strip().upper()
    extracted = ga.cleaning(out)

    if extracted in LABEL_MAPPING:
        if extracted == true_label:
            return "[Add]" + extracted + out, True, False
        return extracted + out, False, False

    if extracted == "E":
        return "[Add]" + out, False, True
    if true_text and true_text.lower() in out.lower():
        return "[Add]" + out + "---" + true_text, True, False
    if "i am not sure" in out.lower():
        return "[Add]" + out, False, True

    return out, False, False


# === Main batch-processing logic ===
def run_task(vc, template, task, diff_mtx):
    data = ga.load_json(os.path.join(MMLU_DIR, f"{task}.json"))
    chars = ga.make_characters(task, TYPE)
    acc = {c: {"correct": 0, "E": 0, "invalid": 0, "total": 0} for c in chars}
    
    with torch.no_grad():  
        for idx, sample in enumerate(tqdm(data, desc=task)):
            ctx = sample["text"]
            true_idx = sample["label"]
            if not 0 <= true_idx < len(LABEL_MAPPING):
                continue
            true_label = LABEL_MAPPING[true_idx]
            true_text = ga.extract_full_correct_text(ctx, true_idx)

            for ch in chars:
                prompt = template.format(character=ch, context=ctx)
                ans = generate_answer_diff(vc, prompt, diff_mtx)

                if ans not in LABEL_MAPPING and ans != "E":
                    ans, is_corr, is_E = handle_invalid_answer_diff(vc, prompt, true_text, true_label, diff_mtx)

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

                ga.update(acc, ch, status)
                acc[ch]["total"] += 1
                sample[f"answer_{ch.replace(' ','_')}"] = ans
    summary = {}
    for ch, c in acc.items():
        pct = (c["correct"] / c["total"]) * 100 if c["total"] else 0
        summary[ch] = {
            "correct": c["correct"],
            "E_count": c["E"],
            "invalid": c["invalid"],
            "total": c["total"],
            "accuracy_percentage": round(pct, 2),
        }
    return data, summary


def main():
    vc = VicundaModel(model_path=MODEL_DIR)
    template = vc.template
    vc.model.eval()

    for alpha, (start, end) in ALPHAS_START_END_PAIRS:
        diff_mtx = build_char_diff(alpha, start, end)
        for task in TASKS:
            print(template)
            print(f"\n=== {task} | α={alpha} | layers={start}–{end} | TOP={TOP} ===")
            data, acc = run_task(vc, template, task, diff_mtx)
            # Save
            save_dir = os.path.join(SAVE_DIR, f"{MODEL}_{alpha}")
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{task}_{SIZE}_answers_{TOP}_{start}_{end}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"data": data, "accuracy": acc}, f, ensure_ascii=False, indent=2)
            for ch, r in acc.items():
                print(
                    f"{ch:>18}: {r['accuracy_percentage']}% "
                    f"(correct {r['correct']}/{r['total']}, "
                    f"E {r['E_count']}, invalid {r['invalid']})"
                )
    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    main()
