#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:41:46 2025

@author: paveenhuang
"""

import json
import argparse
import os
import re
from vicuna import VicundaModel
import random


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


def parse_arguments_and_define_characters():
    """
    Parse command line arguments, split the task, model, size, start, end, and index_str,
    and define the list of characters based on the task.
    index_str can be "133,281,373" -> [133,281,373].
    """
    parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
    parser.add_argument(
        "task_size",
        type=str,
        help=(
            "A combined argument containing: task, model, size, start, end, and neuron_indices. "
            "The neuron_indices can be a comma-separated list. e.g. 'anatomy llama3 8B 1 32 133,281,373'"
        ),
    )
    args = parser.parse_args()

    # Split the combined argument into six parts
    try:
        task, model, size, start, end, index_str = args.task_size.split()
    except ValueError:
        raise ValueError(
            "The task_size parameter should contain six parts: "
            "task, model, size, start, end, and neuron_indices (comma-separated)."
        )

    # Define characters based on the task
    task_name = task.replace("_", " ")
    characters = [f"beginner {task_name}", f"advanced {task_name}"]

    start = int(start)
    end = int(end)

    # 关键修改：将 index_str 解析为多个 neuron index
    neuron_indices = []
    for part in index_str.split(","):
        part = part.strip()
        if part:
            neuron_indices.append(int(part))

    return task, model, size, start, end, neuron_indices, characters


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
    Uses regular expressions to find the first occurrence of A), B), C), or D) and returns the corresponding letter.
    """
    match = re.search(r"\b([A-E])\b", generated_output.upper())
    if match:
        return match.group(1)
    else:
        return generated_output.strip().upper()


def generate_answer(vc, prompt, model):
    """
    Generate an answer using VicundaModel, cleaning the output based on the model type.
    """
    if model.lower() == "phi":
        generated_output = vc.generate([prompt], max_new_tokens=6)[0]
        generated_answer = cleaning(generated_output)
    else:
        generated_answer = vc.generate([prompt], max_new_tokens=1)[0]
    return generated_answer.strip().upper()


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


def update_accuracy_counts(accuracy_counts, character, status):
    """
    Update the accuracy statistics based on the given status (correct, E, invalid).
    """
    if status == "correct":
        accuracy_counts[character]["correct"] += 1
    elif status == "E":
        accuracy_counts[character]["E_count"] += 1
    elif status == "invalid":
        accuracy_counts[character]["invalid"] += 1


def compute_accuracy(accuracy_counts):
    """
    Compute the accuracy for each character.
    """
    accuracy_results = {}
    for character, counts in accuracy_counts.items():
        correct = counts["correct"]
        total = counts["total"]
        E_count = counts["E_count"]
        invalid = counts["invalid"]
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        accuracy_results[character] = {
            "correct": correct,
            "total": total,
            "E_count": E_count,
            "invalid": invalid,
            "accuracy_percentage": round(accuracy, 2),
        }
    return accuracy_results


def save_to_json(data, accuracy_results, save_dir, task, size):
    """
    Save the generated answers and accuracy to a JSON file.
    """
    final_output = {
        "data": data,
        "accuracy": accuracy_results,
    }
    answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers.json")
    print("Saving generated answers and accuracy to JSON...")
    with open(answers_save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"Saved answers and accuracy to {answers_save_path}")


def main():
    # Parse command-line arguments to get model, size, start, end, neuron_indices, and characters (based on the task name)
    # Note: Although the task is passed as an argument, we will ignore it and directly use the TASKS list for all tasks
    task_arg, model_arg, size_arg, start, end, neuron_indices, characters = parse_arguments_and_define_characters()
    
    # Use the passed model and size to define model_path and save_dir (json_path will be constructed for each task in the loop)
    model_path, _, save_dir = define_paths(task_arg, model_arg, size_arg)
    
    # Initialize the VicundaModel
    vc = VicundaModel(model_path=model_path)
    template = vc.template  # Template string
    print("template:", template)
    
    # Initialize overall accuracy counts and aggregated data list
    overall_accuracy_counts = {character: {"correct": 0, "total": 0, "E_count": 0, "invalid": 0} for character in characters}
    all_data = []  # Store the generated results of all tasks
    
    # Iterate over all tasks
    for task in TASKS:
        # Construct the JSON path for each task: assuming each task's JSON file is located at PATH/{task}.json
        json_path_task = os.path.join(PATH, f"{task}.json")
        print(f"\nProcessing task: {task}")
        try:
            task_data = load_json_data(json_path_task)
        except Exception as e:
            print(f"Error loading JSON for task {task}: {e}")
            continue
        
        # Randomly sample 10 examples (if there are fewer than 10, use all)
        if len(task_data) > 10:
            sampled_data = random.sample(task_data, 10)
        else:
            sampled_data = task_data
        
        # Process each sample for the current task
        for idx, sample in enumerate(sampled_data):
            context = sample.get("text", "")
            true_label_int = sample.get("label", -1)
            if true_label_int < 0 or true_label_int >= len(LABEL_MAPPING):
                print(f"Task {task} Sample {idx} has an invalid label: {true_label_int}. Skipping.")
                continue
            true_label = LABEL_MAPPING[true_label_int]
            
            # For each character (e.g., "beginner anatomy", "advanced anatomy")
            for character in characters:
                # Generate the prompt
                prompt = template.format(character=character, context=context)
                # Call generate_lesion to generate an answer, zeroing out neuron_indices in the specified layer range
                outputs = vc.generate_lesion(
                    inputs=[prompt],
                    neuron_indices=neuron_indices,
                    start=start,
                    end=end,
                    max_new_tokens=1,
                    top_p=0.9,
                    temperature=0.0,
                )
                generated_answer = outputs[0].strip().upper()
                
                # Define the key to store the answer (e.g., "answer_beginner_anatomy")
                answer_key = f"answer_{character.replace(' ', '_')}"
                overall_accuracy_counts[character]["total"] += 1
                
                # Check if the generated answer is within A-D
                if generated_answer in LABEL_MAPPING:
                    if generated_answer == true_label:
                        update_accuracy_counts(overall_accuracy_counts, character, "correct")
                elif generated_answer == "E":
                    update_accuracy_counts(overall_accuracy_counts, character, "E")
                else:
                    # Handle invalid answers
                    true_label_text = extract_full_correct_text(context, true_label_int)
                    generated_answer, is_correct, is_E = handle_invalid_answer(vc, prompt, true_label_text, true_label)
                    if is_correct:
                        update_accuracy_counts(overall_accuracy_counts, character, "correct")
                        print(f"[{task}][{idx}][{character}] '{generated_answer}' contains '{true_label_text}' -> Correct")
                    elif is_E:
                        update_accuracy_counts(overall_accuracy_counts, character, "E")
                        print(f"[{task}][{idx}][{character}] '{generated_answer}' -> E")
                    else:
                        update_accuracy_counts(overall_accuracy_counts, character, "invalid")
                        print(f"Task {task}, Sample {idx}, Character '{character}': Invalid generated answer '{generated_answer}'")
                
                # Store the generated answer in the sample dictionary
                sample[answer_key] = generated_answer
            
            # Add task information to the sample and append to the overall data list
            sample["task"] = task
            all_data.append(sample)
        
        print(f"Finished processing task: {task}, {len(sampled_data)} samples processed.")
    
    # Compute overall accuracy and E ratio
    overall_accuracy_results = compute_accuracy(overall_accuracy_counts)
    for character, res in overall_accuracy_results.items():
        print(f"Overall Accuracy for {character}: {res['accuracy_percentage']}% ({res['correct']}/{res['total']})")
        print(f"Overall Number of 'E' answers for {character}: {res['E_count']}")
        print(f"Overall Number of invalid answers for {character}: {res['invalid']}")
    
    # Save all tasks' generated answers and statistical results to a single JSON file
    save_to_json(all_data, overall_accuracy_results, save_dir, "all_tasks", size_arg)
    print("All answers and overall accuracy have been saved successfully.")


if __name__ == "__main__":
    main()
