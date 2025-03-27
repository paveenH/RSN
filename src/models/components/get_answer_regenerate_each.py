#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:57:30 2025

@author: paveenhuang
"""

import os
import argparse
import numpy as np
import json

from vicuna import VicundaModel
import get_answer as ga


LABEL_MAPPING = ["A", "B", "C", "D"]


def parse_arguments_and_define_characters():
    """
    Parse command line arguments, split the task, model, size, top, alpha, start, and end,
    and define the list of characters based on the task.
    """
    parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
    parser.add_argument("task_size", type=str,
                        help="The task, model, size, top, alpha, start, and end as a combined argument, separated by spaces.")
    args = parser.parse_args()

    # Split the combined argument into task, model, size, top, alpha, start, end
    try:
        task, model, size, top, alpha, start, end = args.task_size.split()
    except ValueError:
        raise ValueError(
            "The task_size parameter should contain exactly seven parts: "
            "task, model, size, top, alpha, start, and end, separated by spaces."
        )

    # Define characters based on the task
    # task_name = task.replace('_', ' ')
    # characters = [f"none {task_name}", task_name]
    characters = ["no role"]

    return task, model, size, int(top), characters, float(alpha), int(start), int(end)


def regenerate_answer(vc, prompt, model, char_differences):
    """
    Generate an answer using VicundaModel, cleaning the output based on the model type.
    """
    if model.lower() == "phi":
        generated_output = vc.regenerate([prompt],
                                         diff_matrices=char_differences,
                                         max_new_tokens=6)[0]
        generated_answer = ga.cleaning(generated_output)
    else:
        generated_answer = vc.regenerate([prompt],
                                         diff_matrices=char_differences,
                                         max_new_tokens=1)[0]
    return generated_answer.strip().upper()


def handle_invalid_answer(vc: VicundaModel,
                          prompt: str,
                          true_label_text: str,
                          true_label: str,
                          diff_matrices: np.ndarray,
                          max_new_tokens: int = 8) -> tuple[str, bool, bool]:
    """
    Handle invalid generated answers by re-generating a longer output and checking if it
    contains the correct answer text. Attempts to extract a valid answer using the cleaning logic.
    Return (final_answer, is_correct, is_E).
    """
    # Generate a longer output
    generated_output_long = vc.regenerate(
        [prompt],
        diff_matrices=diff_matrices,
        max_new_tokens=max_new_tokens
    )[0]
    generated_answer = generated_output_long.strip()
    
    # Apply cleaning to extract a potential valid answer
    extracted_answer = ga.cleaning(generated_answer)
    
    # Check if the extracted answer is valid
    if extracted_answer == true_label:
        return "[Add]" + extracted_answer + " original:" + generated_answer, True, False
    
    elif true_label_text and (true_label_text.lower() in generated_answer.lower()):
        return "[Add]" + generated_answer, True, False
    
    elif extracted_answer == 'E' or 'i am not sure' in generated_answer.lower():
        return "[Add]" + generated_answer, False, True
    
    # If no valid answer is found, return as invalid
    return generated_answer, False, False


def save_to_json(data, accuracy_results, save_dir, task, size, top, start, end):
    """
    Save the generated answers and accuracy to a JSON file.
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


def main():
    # 1) Parse and split the arguments
    task, model_name, size, top, characters, alpha, start, end = parse_arguments_and_define_characters()

    # 2) Define paths
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    json_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    matrix_path = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3/{model_name}"
    save_dir = os.path.join(f"/data2/paveen/RolePlaying/src/models/components/answer_modified_v3_each/{model_name}_{alpha}")
    os.makedirs(save_dir, exist_ok=True)

    # 3) Load hs for each task
    none_file = os.path.join(matrix_path, f"none_{task}_{task}_{size}.npy") 
    char_file = os.path.join(matrix_path, f"{task}_{task}_{size}.npy")

    data_none_char = np.load(none_file)  # shape: (samples, 1, layers, hidden_size)
    data_char = np.load(char_file)       # shape: (samples, 1, layers, hidden_size)
    
    # 4) Take mean over 'samples' dimension => shape: (1, layers, hidden_size)
    data_none_char_mean = data_none_char.mean(axis=0)
    data_char_mean      = data_char.mean(axis=0)  
    
    # 5) Compute differences => shape: (1, layers, hidden_size)
    char_differences = data_char_mean - data_none_char_mean
    
    char_differences = char_differences.squeeze(0)  # shape: (layers, hidden_size)
    num_layers = char_differences.shape[0]

    print(f"data_char_mean shape: {data_char_mean.shape}")
    print(f"data_none_char_mean shape: {data_none_char_mean.shape}")
    print(f"char_differences shape(before removing anything): {char_differences.shape}")

    # Debug: Show user
    print(f"layers start from index 0 to {num_layers-1}, but we only apply top logic in [{start}, {end})")

    # 7) Top-N filtering on the specified [start, end) layer range
    if top >= 0:
        print(f"Top {top} calculation begin.")
        for layer_idx in range(num_layers):
            if start <= layer_idx < end:
                layer_diff = char_differences[layer_idx]  # (hidden_size,)
                top_indices = np.argsort(np.abs(layer_diff))[-top:]
                mask = np.zeros_like(layer_diff, dtype=bool)
                mask[top_indices] = True
                char_differences[layer_idx] = np.where(mask, layer_diff, 0)
            else:
                # If outside [start, end), set them to 0
                char_differences[layer_idx] = 0.0

    char_differences = char_differences[1:]  # remove embedding layer index=0
    char_differences = char_differences * alpha
    print(f"char_differences shape after top-{top} masking & removing layer 0: {char_differences.shape}")

    # 9) Initialize the model
    vc = VicundaModel(model_path=model_path)
    template = vc.template  # Assume template is a property of the model

    # Load the MMLU data
    data = ga.load_json_data(json_path)

    # 10) Initialize accuracy counts
    accuracy_counts = {
        character: {"correct": 0, "total": 0, "E_count": 0, "invalid": 0}
        for character in characters
    }

    print("Starting answer generation and accuracy calculation...")

    # 11) Iterate over each sample
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        true_label_int = sample.get("label", -1)
        true_label = LABEL_MAPPING[true_label_int]

        for character in characters:
            prompt = template.format(character=character, context=context)
            generated_answer = regenerate_answer(vc, prompt, model_name, char_differences)

            # Store the answer key
            answer_key = f"answer_{character.replace(' ', '_')}"
            sample[answer_key] = generated_answer
            accuracy_counts[character]["total"] += 1

            # Check the answer
            if generated_answer in LABEL_MAPPING:
                if generated_answer == true_label:
                    ga.update_accuracy_counts(accuracy_counts, character, "correct")
            elif generated_answer == "E":
                ga.update_accuracy_counts(accuracy_counts, character, "E")
            else:
                # Handle invalid answer
                true_label_text = ga.extract_full_correct_text(context, true_label_int)
                regenerated_ans, is_correct, is_E = handle_invalid_answer(
                    vc=vc,
                    prompt=prompt,
                    true_label_text=true_label_text,
                    true_label=true_label,
                    diff_matrices=char_differences,
                    max_new_tokens=8
                )
                sample[answer_key] = regenerated_ans  # Update final answer in sample

                if is_correct:
                    ga.update_accuracy_counts(accuracy_counts, character, "correct")
                    print(f"[{idx}][{character}] => Correct after longer generation.")
                elif is_E:
                    ga.update_accuracy_counts(accuracy_counts, character, "E")
                    print(f"[{idx}][{character}] => Answer is 'E' after longer generation.")
                else:
                    ga.update_accuracy_counts(accuracy_counts, character, "invalid")
                    print(f"Sample {idx}, Character '{character}': Invalid final answer '{regenerated_ans}'")

    # 12) Compute accuracy
    accuracy_results = ga.compute_accuracy(accuracy_counts)

    # Print accuracy results
    for character, results in accuracy_results.items():
        print(f"Accuracy for {character}: {results['accuracy_percentage']}% "
              f"({results['correct']}/{results['total']})")
        print(f"Number of 'E' answers for {character}: {results['E_count']}")
        print(f"Number of invalid answers for {character}: {results['invalid']}")

    # 13) Save the results to JSON (now passing start and end)
    save_to_json(data, accuracy_results, save_dir, task, size, top, start, end)

    print("All answers and accuracy have been saved successfully.")


if __name__ == "__main__":
    main()