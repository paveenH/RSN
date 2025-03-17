#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:41:46 2025

@author: paveenhuang
"""

import json
import os
import re
import random
from vicuna import VicundaModel

# Define constant paths
PATH = "/data2/paveen/RolePlaying/src/models/components/mmlu"
SAVE_BASE_DIR = "/data2/paveen/RolePlaying/src/models/components/answer_lesion"

# Label mapping
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


def define_paths(task, model, size):
    """
    Define model path, JSON data path, and save directory.
    """
    model_path = f"/data2/paveen/RolePlaying/shared/{model}/{size}"
    json_path = os.path.join(PATH, f"{task}.json")
    save_dir = os.path.join(SAVE_BASE_DIR, model)
    os.makedirs(save_dir, exist_ok=True)
    return model_path, json_path, save_dir


def load_json_data(json_path):
    """
    Load JSON data.
    """
    print(f"Loading JSON data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total samples loaded: {len(data)}")
    return data


def extract_full_correct_text(question_text, label_index):
    """
    Extract the complete sentence corresponding to the given label (A/B/C/D) from the question text.
    """
    lines = question_text.split("\n")
    option_letters = ["A", "B", "C", "D"]
    prefix = f"{option_letters[label_index]})"
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.upper().startswith(prefix):
            return line_stripped[len(prefix) :].strip().lower()
    return None


def cleaning(generated_output):
    """
    Clean the generated output to extract the answer option (A, B, C, D).
    Uses regular expressions to find the first occurrence of A-E and returns the corresponding letter.
    """
    match = re.search(r"\b([A-E])\b", generated_output.upper())
    if match:
        return match.group(1)
    else:
        return generated_output.strip().upper()


def handle_invalid_answer(vc, prompt, true_label_text, true_label):
    """
    Handle invalid generated answers by re-generating a longer output and checking if it contains the correct answer text.
    Attempts to extract a valid answer using the cleaning logic.
    """
    # Generate a longer output
    generated_output_long = vc.generate([prompt], max_new_tokens=8)[0]
    generated_answer = generated_output_long.strip()

    # Apply cleaning to extract a potential valid answer
    extracted_answer = cleaning(generated_answer)

    # Check if the extracted answer is valid
    if extracted_answer == true_label:
        return "[Add]" + extracted_answer + " original:" + generated_answer, True, False

    # Fallback: Check if the correct answer text is contained in the generated output
    elif true_label_text and true_label_text.lower() in generated_answer.lower():
        return "[Add]" + generated_answer, True, False

    elif extracted_answer == "E" or "i am not sure" in generated_answer.lower():
        return "[Add]" + generated_answer, False, True

    # If no valid answer is found, return the output as invalid
    return generated_answer, False, False


def update_accuracy_counts(accuracy_counts, overall_key, status):
    """
    Update the accuracy statistics based on the given status (correct, E, invalid)
    for the overall_key (i.e., 'none' or 'expert').
    """
    if status == "correct":
        accuracy_counts[overall_key]["correct"] += 1
    elif status == "E":
        accuracy_counts[overall_key]["E_count"] += 1
    elif status == "invalid":
        accuracy_counts[overall_key]["invalid"] += 1


def compute_accuracy(accuracy_counts):
    """
    Compute the accuracy for each overall key (e.g., 'none', 'expert').
    """
    accuracy_results = {}
    for key, counts in accuracy_counts.items():
        correct = counts["correct"]
        total = counts["total"]
        E_count = counts["E_count"]
        invalid = counts["invalid"]
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        accuracy_results[key] = {
            "correct": correct,
            "total": total,
            "E_count": E_count,
            "invalid": invalid,
            "accuracy_percentage": round(accuracy, 2),
        }
    return accuracy_results


def save_to_json(data, accuracy_results, save_dir, task, size, index):
    """
    Save the generated answers and accuracy to a JSON file.
    """
    final_output = {
        "data": data,
        "accuracy": accuracy_results,
    }
    answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers_{index}.json")
    print("Saving generated answers and accuracy to JSON...")
    with open(answers_save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"Saved answers and accuracy to {answers_save_path}")


def map_char_to_overall(character: str) -> str:
    """
    Map the full prompt character, e.g. "none anatomy" or "anatomy", to an overall key: "none" or "expert".
    """
    if character.lower().startswith("none"):
        return "none"
    else:
        return "expert"


def main():
    # 1) Hardcode the model, size, layer range, and neuron_indices for demonstration
    model_arg = "llama3"
    size_arg = "8B"
    start = 1
    end = 31
    neuron_indices = "2629"
    
    # Print the parameters for debugging/logging
    print("Running with parameters:")
    print(f"  Model: {model_arg}")
    print(f"  Size: {size_arg}")
    print(f"  Layer range: {start} to {end}")
    print(f"  Neuron indices: {neuron_indices}")

    # 2) Build model_path
    model_path = f"/data2/paveen/RolePlaying/shared/{model_arg}/{size_arg}"
    save_dir = f"/data2/paveen/RolePlaying/src/models/components/answer_lesion/{model_arg}"
    os.makedirs(save_dir, exist_ok=True)

    # 3) Initialize the model
    vc = VicundaModel(model_path=model_path)
    template = vc.template
    print("template:", template)

    # 4) We only need overall keys: "none" and "expert"
    #    So we define them in overall_accuracy_counts
    overall_accuracy_counts = {
        "none":  {"correct": 0, "total": 0, "E_count": 0, "invalid": 0},
        "expert": {"correct": 0, "total": 0, "E_count": 0, "invalid": 0},
    }
    all_data = []

    # 5) Iterate over tasks
    for task in TASKS:
        task_name = task.replace("_", " ")
        # build the characters for prompts
        # e.g. ["none astronomy", "astronomy"]
        prompt_characters = [f"none {task_name}", task_name]

        # define paths
        _, json_path_task, _ = define_paths(task, model_arg, size_arg)
        print(f"\nProcessing task: {task}")

        # load data
        try:
            task_data = load_json_data(json_path_task)
        except Exception as e:
            print(f"Error loading JSON for task {task}: {e}")
            continue

        # sample 10
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

            # For each character in [f"none {task_name}", task_name]
            for character in prompt_characters:
                # Build prompt
                prompt = template.format(character=character, context=context)

                # Generate with lesion
                outputs = vc.generate_lesion(
                    inputs=[prompt],
                    neuron_indices=[int(x) for x in neuron_indices.split(",")],
                    start=start,
                    end=end,
                    max_new_tokens=1,
                    top_p=0.9,
                    temperature=0.0,
                )
                gen_ans = outputs[0].strip().upper()

                # Map to overall key
                overall_key = map_char_to_overall(character)

                # Increase total
                overall_accuracy_counts[overall_key]["total"] += 1

                # Check
                if gen_ans in LABEL_MAPPING:
                    if gen_ans == true_label:
                        update_accuracy_counts(overall_accuracy_counts, overall_key, "correct")
                elif gen_ans == "E":
                    update_accuracy_counts(overall_accuracy_counts, overall_key, "E")
                else:
                    # handle invalid
                    true_label_text = extract_full_correct_text(context, true_label_int)
                    new_ans, is_correct, is_E = handle_invalid_answer(vc, prompt, true_label_text, true_label)
                    if is_correct:
                        update_accuracy_counts(overall_accuracy_counts, overall_key, "correct")
                    elif is_E:
                        update_accuracy_counts(overall_accuracy_counts, overall_key, "E")
                    else:
                        update_accuracy_counts(overall_accuracy_counts, overall_key, "invalid")
                    gen_ans = new_ans

                # store the final answer in sample
                answer_key = f"answer_{character.replace(' ', '_')}"
                sample[answer_key] = gen_ans

            sample["task"] = task
            all_data.append(sample)

        print(f"Task {task}: processed {len(sampled_data)} samples.")

    # 6) compute overall accuracy
    overall_accuracy_results = compute_accuracy(overall_accuracy_counts)

    # 7) Print results
    for key, res in overall_accuracy_results.items():
        print(f"Overall Accuracy for {key}: {res['accuracy_percentage']}% ({res['correct']}/{res['total']})")
        print(f"Overall 'E': {res['E_count']}, invalid: {res['invalid']}")

    # 8) Save
    save_to_json(all_data, overall_accuracy_results, save_dir, "all_tasks", size_arg, neuron_indices)
    print("All tasks processed. Results saved.")


if __name__ == "__main__":
    main()