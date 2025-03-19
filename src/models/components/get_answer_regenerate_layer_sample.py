#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:36:29 2025
Modified to process multiple TASKs and save all results in one file.
@author: paveenhuang
"""

import os
import numpy as np
import json
import random
from vicuna import VicundaModel
import get_answer as ga

LABEL_MAPPING = ["A", "B", "C", "D"]

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


def map_char_to_overall(character: str) -> str:
    """
    Map the full prompt description to the overall metric key:
        If it starts with "none", it is mapped to "none"; otherwise, it is mapped to "expert".
        """
    if character.lower().startswith("none"):
        return "none"
    else:
        return "expert"


def regenerate_answer(vc, prompt, model, char_differences):
    """
    Generate an answer using VicundaModel, cleaning the output based on the model type.
    """
    if model.lower() == "phi":
        generated_output = vc.regenerate([prompt],diff_matrices=char_differences, max_new_tokens=6)[0]
        generated_answer = ga.cleaning(generated_output)
    else:
        generated_answer = vc.regenerate(
            [prompt],
            diff_matrices=char_differences, 
            max_new_tokens=1
        )[0]
    return generated_answer.strip().upper()


def handle_invalid_answer(vc: VicundaModel, 
                          prompt: str, 
                          true_label_text: str, 
                          true_label: str,
                          diff_matrices: np.ndarray,
                          max_new_tokens: int = 8) -> tuple[str, bool]:
    """
    Handle invalid generated answers by re-generating a longer output and checking if it contains the correct answer text.
    Attempts to extract a valid answer using the cleaning logic.
    """
    # Generate a longer output
    generated_output_long = vc.regenerate([prompt], diff_matrices=diff_matrices, max_new_tokens=max_new_tokens)[0]
    generated_answer = generated_output_long.strip()
    
    # Apply cleaning to extract a potential valid answer
    extracted_answer = ga.cleaning(generated_answer)
    
    # Check if the extracted answer is valid
    if extracted_answer == true_label:
        return "[Add]" + extracted_answer + " original:" + generated_answer, True, False
    
    # Fallback: Check if the correct answer text is contained in the generated output
    elif true_label_text and true_label_text.lower() in generated_answer.lower():
        return "[Add]" + generated_answer, True, False
    
    elif extracted_answer == 'E' or 'i am not sure' in generated_answer.lower():
        return "[Add]" + generated_answer, False, True
    
    # If no valid answer is found, return the output as invalid
    return generated_answer, False, False


def save_to_json(data, accuracy_results, save_dir, task, size, top, start, end):
    """
    Save the generated answers and accuracy as a JSON file.
    """
    final_output = {
        "data": data,
        "accuracy": accuracy_results,
    }
    answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers_{top}_{start}_{end}.json")
    print("Saving generated answers and accuracy to JSON...")
    with open(answers_save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"Saved answers and accuracy to {answers_save_path}")


def main(top, alpha, start, end):
    model_name = "llama3"
    size = "8B"

    # Define each path
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    mmlu_path = "/data2/paveen/RolePlaying/src/models/components/mmlu"
    matrix_path = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3_mean/{model_name}"
    save_dir = os.path.join(f"/data2/paveen/RolePlaying/src/models/components/answer_sample/{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    # Loading difference matrix
    try:
        data_char_diff = np.load(os.path.join(matrix_path, f'all_mean_{size}.npy'))       # shape: (1,1,layers,hidden_size)
        data_none_char_diff = np.load(os.path.join(matrix_path, f'none_all_mean_{size}.npy')) # shape: (1,1,layers,hidden_size)
    except FileNotFoundError as e:
        print(f"Error loading difference matrices: {e}")
        exit(1)

    # The dissimilarity matrix is calculated and scaled by alpha
    char_differences = (data_char_diff - data_none_char_diff).squeeze(0).squeeze(0)  # shape: (layers, hidden_size)
    num_layers = char_differences.shape[0]
    start = max(0, min(start, num_layers - 1))
    end = max(start + 1, min(end, num_layers))
    
    # Exclude the embedding layer (assuming layer 0 is an embedding) and scale by alpha
    # char_differences = char_differences[1:] * alpha
    char_differences = char_differences * alpha

    # Perform top masking on the specified layer
    if top >= 0:
        print(f"Applying top-{top} masking to char_differences.")
        for layer_idx in range(num_layers):
            if start <= layer_idx < end:
                layer_diff = char_differences[layer_idx]  # shape: (hidden_size,)
                top_indices = np.argsort(np.abs(layer_diff))[-top:]
                mask = np.zeros_like(layer_diff, dtype=bool)
                mask[top_indices] = True
                char_differences[layer_idx] = np.where(mask, layer_diff, 0)
            else:
                char_differences[layer_idx] = 0
    
    char_differences = char_differences[1:]
    print(f"char_differences shape after top masking: {char_differences.shape}")

    # Initializing the model
    vc = VicundaModel(model_path=model_path)
    template = vc.template 
    print("template:", template)

    # Initialize the overall accuracy count (by "none" and "expert")
    overall_accuracy_counts = {
        "none": {"correct": 0, "total": 0, "E_count": 0, "invalid": 0},
        "expert": {"correct": 0, "total": 0, "E_count": 0, "invalid": 0},
    }
    all_data = []

    for task in TASKS:
        task_name = task.replace("_", " ")
        prompt_characters = [f"none {task_name}", task_name]

        json_path = os.path.join(mmlu_path, f"{task}.json")
        print(f"\nProcessing task: {task}")

        try:
            task_data = ga.load_json_data(json_path)
        except Exception as e:
            print(f"Error loading JSON for task {task}: {e}")
            continue

        # Sample
        if len(task_data) > 10:
            sampled_data = random.sample(task_data, 10)
        else:
            sampled_data = task_data

        for idx, sample in enumerate(sampled_data):
            context = sample.get("text", "")
            true_label_int = sample.get("label", -1)
            if true_label_int < 0 or true_label_int >= len(LABEL_MAPPING):
                print(f"Task {task}, sample {idx} invalid label: {true_label_int}")
                continue
            true_label = LABEL_MAPPING[true_label_int]

            for character in prompt_characters:
                prompt = template.format(character=character, context=context)
                generated_answer = regenerate_answer(vc, prompt, model_name, char_differences)

                overall_key = map_char_to_overall(character)
                overall_accuracy_counts[overall_key]["total"] += 1

                if generated_answer in LABEL_MAPPING:
                    if generated_answer == true_label:
                        ga.update_accuracy_counts(overall_accuracy_counts, overall_key, "correct")
                elif generated_answer == "E":
                    ga.update_accuracy_counts(overall_accuracy_counts, overall_key, "E")
                else:
                    true_label_text = ga.extract_full_correct_text(context, true_label_int)
                    generated_answer, is_correct, is_E = handle_invalid_answer(
                        vc=vc,
                        prompt=prompt,
                        true_label_text=true_label_text,
                        true_label=true_label,
                        diff_matrices=char_differences,
                        max_new_tokens=8
                    )
                    if is_correct:
                        ga.update_accuracy_counts(overall_accuracy_counts, overall_key, "correct")
                        print(f"[{idx}][{character}] '{generated_answer}' contains '{true_label_text}' -> Correct")
                    elif is_E:
                        ga.update_accuracy_counts(overall_accuracy_counts, overall_key, "E")
                        print(f"[{idx}][{character}] '{generated_answer}' -> E")
                    else:
                        ga.update_accuracy_counts(overall_accuracy_counts, overall_key, "invalid")
                        print(f"Task {task}, Sample {idx}, Character '{character}': Invalid generated answer '{generated_answer}'")

                answer_key = f"answer_{character.replace(' ', '_')}"
                sample[answer_key] = generated_answer

            sample["task"] = task
            all_data.append(sample)

        print(f"Task {task}: processed {len(sampled_data)} samples.")

    overall_accuracy_results = ga.compute_accuracy(overall_accuracy_counts)

    for key, res in overall_accuracy_results.items():
        print(f"Overall Accuracy for {key}: {res['accuracy_percentage']}% ({res['correct']}/{res['total']})")
        print(f"Overall 'E': {res['E_count']}, invalid: {res['invalid']}")

    save_to_json(all_data, overall_accuracy_results, save_dir, "all_tasks", size, top, start, end)
    print("All tasks processed. Results saved.")

if __name__ == "__main__":
    start = 1
    end = 31
    alpha = 1
    for top in (5, 5, 51):
        main(top, alpha, start, end)