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


def main():
    task, model, size, start, end, neuron_indices, characters = parse_arguments_and_define_characters()
    model_path, json_path, save_dir = define_paths(task, model, size)

    # Initialize the model
    vc = VicundaModel(model_path=model_path)
    template = vc.template  # Assume template is a property of the model
    print("template:", template)

    # Load the data
    data = load_json_data(json_path)

    # Sample 10 random samples from the loaded data
    if len(data) > 10:
        data = random.sample(data, 10)

    # Initialize accuracy counts
    accuracy_counts = {character: {"correct": 0, "total": 0, "E_count": 0, "invalid": 0} for character in characters}

    print("Starting answer generation and accuracy calculation...")

    # Iterate over each sample
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        true_label_int = sample.get("label", -1)
        if true_label_int < 0 or true_label_int >= len(LABEL_MAPPING):
            print(f"Sample {idx} has an invalid label: {true_label_int}. Skipping.")
            continue
        true_label = LABEL_MAPPING[true_label_int]

        for character in characters:
            # Generate the prompt
            prompt = template.format(character=character, context=context)

            outputs = vc.generate_lesion(
                inputs=[prompt],
                neuron_indices=neuron_indices,  # 置零的 neuron index 列表
                start=start,
                end=end,
                max_new_tokens=1,
                top_p=0.9,
                temperature=0.0,
            )

            generated_answer = outputs[0].strip().upper()

            # Store the answer key
            answer_key = f"answer_{character.replace(' ', '_')}"
            accuracy_counts[character]["total"] += 1

            # Check the answer
            if generated_answer in LABEL_MAPPING:
                if generated_answer == true_label:
                    update_accuracy_counts(accuracy_counts, character, "correct")
            elif generated_answer == "E":
                update_accuracy_counts(accuracy_counts, character, "E")
            else:
                # Handle invalid answer
                true_label_text = extract_full_correct_text(context, true_label_int)
                generated_answer, is_correct, is_E = handle_invalid_answer(vc, prompt, true_label_text, true_label)
                if is_correct:
                    update_accuracy_counts(accuracy_counts, character, "correct")
                    print(f"[{idx}][{character}] '{generated_answer}' contains '{true_label_text}' -> Correct")
                elif is_E:
                    update_accuracy_counts(accuracy_counts, character, "E")
                    print(f"[{idx}][{character}] '{generated_answer}' -> E")
                else:
                    update_accuracy_counts(accuracy_counts, character, "invalid")
                    print(f"Sample {idx}, Character '{character}': Invalid generated answer '{generated_answer}'")

            # Store the generated answer
            sample[answer_key] = generated_answer

    # Compute accuracy
    accuracy_results = compute_accuracy(accuracy_counts)

    # Print accuracy results
    for character, results in accuracy_results.items():
        print(f"Accuracy for {character}: {results['accuracy_percentage']}% ({results['correct']}/{results['total']})")
        print(f"Number of 'E' answers for {character}: {results['E_count']}")
        print(f"Number of invalid answers for {character}: {results['invalid']}")

    # Save the results to JSON
    save_to_json(data, accuracy_results, save_dir, task, size, neuron_indices)

    print("All answers and accuracy have been saved successfully.")


if __name__ == "__main__":
    main()
