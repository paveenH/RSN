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

from llms import VicundaModel
import get_answer_alltasks as ga

LABEL_MAPPING = ["A", "B", "C", "D"]

# === Configuration ===
TASKS = ga.TASKS

MODEL = "hermes"
SIZE = "3B"
TOP = 15
ALPHAS = [4]
START_END_PAIRS = [(7, 14), (7, 17)]

SHORT = 1
LONG = 10

DIFFUSION = None  # dream/ llada/ None
STEP = 16

MODEL_DIR = "NousResearch/Hermes-3-Llama-3.2-3B"
MAT_DIR = f"/data2/paveen/RolePlaying/components/hidden_states_v3_mean/{MODEL}"
JSON_DIR = "/data2/paveen/RolePlaying/components/mmlu"

def make_characters(task_name: str):
    task_name = task_name.replace("_", " ")
    return [
        f"non-{task_name}",
        f"{task_name}",
    ]


# === Helper functions ===
def compute_accuracy(acc_dict):
    out = {}
    for ch, d in acc_dict.items():
        pct = (d["correct"] / d["total"]) * 100 if d["total"] else 0
        out[ch] = {
            **d,
            "accuracy_percentage": round(pct, 2),
        }
    return out


def regenerate_answer(vc, prompt, char_differences, diffusion_mode=None):
    out = vc.regenerate([prompt], 
                        diff_matrices=char_differences, 
                        max_new_tokens=SHORT)[0]
    return ga.cleaning(out)


def handle_invalid_answer(vc, prompt, true_text, true_label, diff_matrices, max_new_tokens=LONG):
    """
    Retry with a longer output if the first answer was invalid.
    Returns (formatted_answer, is_correct, is_E).
    """
    out_long = vc.regenerate([prompt], diff_matrices=diff_matrices, max_new_tokens=max_new_tokens)[0].strip()
    out_long = out_long.replace("<|assistant|>", "").replace("\u200b", "").strip().upper()
    extracted = ga.cleaning(out_long)
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


def save_to_json(data, accuracy_results, save_dir, task, size, top, start, end):
    """
    Save answers and accuracy into a JSON file under save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    out = {"data": data, "accuracy": accuracy_results}
    fname = f"{task}_{size}_answers_{top}_{start}_{end}.json"
    path = os.path.join(save_dir, fname)
    print(f"Saving results to {path} ...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)


# === Main batch-processing logic ===
def get_diff_matrix(matrix_dir, start, end):
    try:
        data_char = np.load(os.path.join(matrix_dir, f"diff_mean_{SIZE}.npy"))
        data_none = np.load(os.path.join(matrix_dir, f"none_diff_mean_{SIZE}.npy"))
    except FileNotFoundError as e:
        print(f"[ERROR] Missing diff matrix: {e}")

    diff = (data_char - data_none).squeeze(0).squeeze(0)
    num_layers, hidden_size = diff.shape

    for layer in range(num_layers):
        if start <= layer < end:
            layer_diff = diff[layer]
            idxs = np.argsort(np.abs(layer_diff))[-TOP:]
            mask = np.zeros_like(layer_diff, dtype=bool)
            mask[idxs] = True
            diff[layer] = layer_diff * mask
        else:
            diff[layer] = 0
    return diff
        
    
def main():
    vc = VicundaModel(model_path=MODEL_DIR)
    vc.model.eval()
    template = vc.template
    
    for task in TASKS:
        print(template)
        for alpha in ALPHAS:
            for start, end in START_END_PAIRS:
                print(f"\n[RUNNING] task={task}, top={TOP}, α={alpha}, layers={start}-{end}")

                characters = make_characters(task)
                
                diff = get_diff_matrix(MAT_DIR, start, end)
                char_diff = diff[1:] * alpha
                
                data = ga.load_json(os.path.join(JSON_DIR, f"{task}.json"))
                accuracy_counts = {c: {"correct": 0, "total": 0, "E_count": 0, "invalid": 0} for c in characters}

                for idx, sample in enumerate(data):
                    context = sample.get("text", "")
                    true_int = sample.get("label", -1)
                    true_label = LABEL_MAPPING[true_int]
                    full_text = ga.extract_full_correct_text(context, true_int)

                    for char in characters:
                        prompt = template.format(character=char, context=context)
                        ans = regenerate_answer(vc, prompt, char_diff)
                        key = f"answer_{char.replace(' ', '_')}"
                        sample[key] = ans
                        accuracy_counts[char]["total"] += 1

                        if ans in LABEL_MAPPING:
                            if ans == true_label:
                                ga.update(accuracy_counts, char, "correct")
                        elif ans == "E":
                            ga.update(accuracy_counts, char, "E")
                        else:
                            ans2, corr, isE = handle_invalid_answer(vc, prompt, full_text, true_label, diff_matrices=char_diff)
                            sample[key] = ans2
                            if corr:
                                ga.update(accuracy_counts, char, "correct")
                                print(f"[FIXED][{idx}][{char}] {ans2}")
                            elif isE:
                                ga.update(accuracy_counts, char, "E")
                            else:
                                ga.update(accuracy_counts, char, "invalid")
                                print(f"[INVALID][{idx}][{char}] {ans2}")

                results = compute_accuracy(accuracy_counts)
                for char, stats in results.items():
                    print(
                        f"  -> {char}: {stats['accuracy_percentage']}% "
                        f"({stats['correct']}/{stats['total']}), "
                        f"E_count={stats['E_count']}, invalid={stats['invalid']}"
                    )

                save_dir = f"/data2/paveen/RolePlaying/src/models/components/answer_mdf/{MODEL}_{alpha}"
                save_to_json(data, results, save_dir, task, SIZE, TOP, start, end)

    print("\n[ALL DONE] All tasks have been processed.")


if __name__ == "__main__":
    main()
