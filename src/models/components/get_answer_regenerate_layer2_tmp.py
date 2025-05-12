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

from vicuna import VicundaModel
import get_answer as ga

LABEL_MAPPING = ["A", "B", "C", "D"]

# === Configuration: define all combinations here ===
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

# MODELS = "phi"
# SIZES = "3.8"
# TOPS = 15
# ALPHAS = [ 5, 10, 15]
# START_END_PAIRS = [(13, 24)]
# NUM_GPUS = 1

# MODELS = "qwen2.5"
# SIZES = "0.5B"
# TOPS = 4
# ALPHAS = [3,7]
# START_END_PAIRS = [(9, 16)]
# NUM_GPUS = 1

MODELS = "qwen2.5"
SIZES = "3B"
TOPS = 4
ALPHAS = [3,5,7]
START_END_PAIRS = [(21, 31)]
NUM_GPUS = 1

# MODELS = "llama3"
# SIZES = "1B"
# TOPS = 10
# ALPHAS = [ 1,2,3,4]
# START_END_PAIRS = [(4, 9)]
# NUM_GPUS = 1

# === Helper functions (as in your original script) ===

def regenerate_answer(vc, prompt, model, char_differences):
    """
    Generate an answer using VicundaModel, cleaning the output based on the model type.
    """
    if model.lower() == "phi":
        out = vc.regenerate([prompt], diff_matrices=char_differences, max_new_tokens=6)[0]
        return ga.cleaning(out).strip().upper()
    else:
        out = vc.regenerate(
            [prompt],
            diff_matrices=char_differences,
            max_new_tokens=1
        )[0]
        return out.strip().upper()

def handle_invalid_answer(vc, prompt, true_label_text, true_label,
                          diff_matrices, max_new_tokens=8):
    """
    Retry with a longer output if the first answer was invalid.
    Returns (formatted_answer, is_correct, is_E).
    """
    long_out = vc.regenerate([prompt], diff_matrices=diff_matrices,
                             max_new_tokens=max_new_tokens)[0].strip()
    extracted = ga.cleaning(long_out)
    # exact match
    if extracted == true_label:
        return "[Add]" + extracted + " original:" + long_out, True, False
    # contains the correct text
    if true_label_text and true_label_text.lower() in long_out.lower():
        return "[Add]" + long_out, True, False
    # maps to E or uncertainty
    if extracted == 'E' or 'i am not sure' in long_out.lower():
        return "[Add]" + long_out, False, True
    return long_out, False, False

def save_to_json(data, accuracy_results, save_dir,
                 task, size, top, start, end):
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
def main():
    model_name = MODELS  # "llama3"
    size = SIZES         # "8B"
    top = TOPS

    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    print(f"[INFO] Loading model {model_name}/{size} from {model_path}")
    vc = VicundaModel(model_path=model_path, num_gpus=NUM_GPUS)
    template = vc.template

    matrix_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_mean/{model_name}"
    json_dir = "/data2/paveen/RolePlaying/src/models/components/mmlu"

    for task in TASKS:
        task_name = task.replace('_', ' ')

        for alpha in ALPHAS:
            for start, end in START_END_PAIRS:
                print(f"\n[RUNNING] task={task}, top={top}, α={alpha}, layers={start}-{end}")
                characters = [f"none {task_name}", task_name]

                try:
                    data_char = np.load(os.path.join(matrix_dir, f"diff_mean_{size}.npy"))
                    data_none = np.load(os.path.join(matrix_dir, f"none_diff_mean_{size}.npy"))
                except FileNotFoundError as e:
                    print(f"[ERROR] Missing diff matrix: {e}")
                    continue

                diff = (data_char - data_none).squeeze(0).squeeze(0)
                num_layers, hidden_size = diff.shape

                if top >= 0:
                    for layer in range(num_layers):
                        if start <= layer < end:
                            layer_diff = diff[layer]
                            idxs = np.argsort(np.abs(layer_diff))[-top:]
                            mask = np.zeros_like(layer_diff, dtype=bool)
                            mask[idxs] = True
                            diff[layer] = layer_diff * mask
                        else:
                            diff[layer] = 0

                char_diff = diff[1:] * alpha
                json_path = os.path.join(json_dir, f"{task}.json")
                data = ga.load_json_data(json_path)

                accuracy_counts = {
                    c: {"correct": 0, "total": 0, "E_count": 0, "invalid": 0}
                    for c in characters
                }

                for idx, sample in enumerate(data):
                    context = sample.get("text", "")
                    true_int = sample.get("label", -1)
                    true_label = LABEL_MAPPING[true_int]

                    for char in characters:
                        prompt = template.format(character=char, context=context)
                        ans = regenerate_answer(vc, prompt, model_name, char_diff)
                        key = f"answer_{char.replace(' ', '_')}"
                        sample[key] = ans
                        accuracy_counts[char]["total"] += 1

                        if ans in LABEL_MAPPING:
                            if ans == true_label:
                                ga.update_accuracy_counts(accuracy_counts, char, "correct")
                        elif ans == "E":
                            ga.update_accuracy_counts(accuracy_counts, char, "E")
                        else:
                            full_text = ga.extract_full_correct_text(context, true_int)
                            ans2, corr, isE = handle_invalid_answer(vc, prompt, full_text, true_label, diff_matrices=char_diff)
                            sample[key] = ans2
                            if corr:
                                ga.update_accuracy_counts(accuracy_counts, char, "correct")
                                print(f"[FIXED][{idx}][{char}] {ans2}")
                            elif isE:
                                ga.update_accuracy_counts(accuracy_counts, char, "E")
                            else:
                                ga.update_accuracy_counts(accuracy_counts, char, "invalid")
                                print(f"[INVALID][{idx}][{char}] {ans2}")

                results = ga.compute_accuracy(accuracy_counts)
                for char, stats in results.items():
                    print(f"  -> {char}: {stats['accuracy_percentage']}% "
                          f"({stats['correct']}/{stats['total']}), "
                          f"E_count={stats['E_count']}, invalid={stats['invalid']}")

                save_dir = f"/data2/paveen/RolePlaying/src/models/components/answer_mdf/{model_name}_{alpha}"
                save_to_json(data, results, save_dir, task, size, top, start, end)
            

    print("\n[ALL DONE] All tasks have been processed.")

if __name__ == "__main__":
    main()