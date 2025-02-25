#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:18:18 2025

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
    parser.add_argument("task_size", type=str, help="The task, model, size, top, alpha, start, and end as a combined argument, separated by spaces.")
    args = parser.parse_args()

    parts = args.task_size.split()
    if len(parts) == 7:
        task, model, size, top, alpha, start, end = parts
        index_list = ""  
    elif len(parts) == 8:
        task, model, size, top, alpha, start, end, index_list_str = parts
        index_list = [int(x) for x in index_list_str.split(',')]
    else:
        raise ValueError("The task_size parameter should contain either 7 or 8 parts: task, model, size, top, alpha, start, end, [optional index_list].")


    # Define characters based on the task
    task_name = task.replace('_', ' ')
    characters = [f"none {task_name}", task_name] 

    return task, model, size, int(top), characters, float(alpha), int(start), int(end), index_list


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


def save_to_json(data, accuracy_results, save_dir, task, size, top, start, end, index_list=None):
    """
    Save the generated answers and accuracy to a JSON file.
    """
    final_output = {"data": data, "accuracy": accuracy_results}
    if index_list:
        answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers_{top}_{start}_{end}_{len(index_list)}.json")
    else:
        answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers_{top}_{start}_{end}.json")
    print("Saving generated answers and accuracy to JSON...")
    with open(answers_save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"Saved answers and accuracy to {answers_save_path}")

def get_difference_matrix(data_char_diff, data_none_char_diff, start, end, alpha, top, top_overall):
    # Compute the difference matrix and apply scaling with alpha
    char_differences = (data_char_diff - data_none_char_diff).squeeze(0).squeeze(0)    # (layers,hidden_size)
    
    # Ensure start and end are in range
    num_layers = char_differences.shape[0]
    start = max(0, min(start, num_layers - 1))
    end = max(start + 1, min(end, num_layers))
    
    char_differences = char_differences[1:] * alpha # Exclude the embedding layer
    
    # Debug
    print(f"data_char_diff shape: {data_char_diff.shape}")
    print(f"data_none_char_diff shape: {data_none_char_diff.shape}")
    print(f"char_differences shape: {char_differences.shape}")
    print(f"layers start from {start} to {end}")
    
    # Only in the start to end layers are counted
    top_indices_list = []
    for layer_idx in range(start, end):
        layer_diff = char_differences[layer_idx]  # layer_diff shape: (hidden_size,)
        top_indices = np.argsort(np.abs(layer_diff))[-top:]
        top_indices_list.append(top_indices)

    top_indices_matrix = np.array(top_indices_list)

    flat_top_indices = top_indices_matrix.flatten()
    unique_indices, counts = np.unique(flat_top_indices, return_counts=True)
    sorted_order = np.argsort(-counts)
    unique_indices_sorted = unique_indices[sorted_order]

    significant_neurons = set(unique_indices_sorted[:top_overall])
    print("Significant neurons (top overall in layers {}-{}): {}".format(start, end, significant_neurons))

    num_layers_modified = char_differences.shape[0]
    for layer_idx in range(num_layers_modified):
        if start <= layer_idx < end:
            layer_diff = char_differences[layer_idx]  # shape: (hidden_size,)
            mask = np.isin(np.arange(layer_diff.shape[0]), list(significant_neurons))
            char_differences[layer_idx] = np.where(mask, layer_diff, 0)
        else:
            char_differences[layer_idx] = 0
    
    return char_differences

def get_difference_matrix_ablation(
    data_char_diff: np.ndarray,
    data_none_char_diff: np.ndarray,
    start: int,
    end: int,
    alpha: float,
    top: int,
    ablation_indices: list
) -> np.ndarray:
    """
    Similar to get_difference_matrix, but performs "whole-column random replacement" for the neuron indices
    specified in ablation_indices to evaluate the impact of these high-frequency neurons.

    :param data_char_diff:       shape (1,1,layers,hidden_size)
    :param data_none_char_diff:  shape (1,1,layers,hidden_size)
    :param start:                Starting layer index (excluding the embedding layer, which is automatically offset here)
    :param end:                  Ending layer index
    :param alpha:                Scaling factor
    :param top:                  Number of top neurons to keep based on absolute values per layer
    :param ablation_indices:     List/set of neuron indices to perform random replacement
    :return:                     Difference matrix of shape (layers, hidden_size)
    """
    char_differences = (data_char_diff - data_none_char_diff).squeeze(0).squeeze(0)  # (layers, hidden_size)
    hidden_size = char_differences.shape[1]

    # Exclude the embedding layer (the 0th layer), and multiply by alpha
    char_differences = char_differences[1:] * alpha  # shape (layers-1, hidden_size)
    num_layers_modified = char_differences.shape[0]

    # Correct the start/end range to ensure it doesn't go out of bounds
    start = max(0, min(start, num_layers_modified - 1))
    end = max(start + 1, min(end, num_layers_modified))

    print(f"[Ablation] data_char_diff shape: {data_char_diff.shape}")
    print(f"[Ablation] data_none_char_diff shape: {data_none_char_diff.shape}")
    print(f"[Ablation] char_differences shape after excluding embedding layer: {char_differences.shape}")
    print(f"[Ablation] layers range: [{start}, {end})")
    print(f"[Ablation] index: {ablation_indices}")

    # Save the value for random copy
    layer_diff_original = [None] * num_layers_modified
    for layer_idx in range(num_layers_modified):
        layer_diff_original[layer_idx] = char_differences[layer_idx].copy()

    # Count frequence from start to end
    top_indices_list = []
    non_top_indices_list = []
    for layer_idx in range(start, end):
        layer_diff = char_differences[layer_idx]  # shape (hidden_size,)
        top_indices = np.argsort(np.abs(layer_diff))[-top:]  # Select top neurons with the highest absolute values
        top_indices_list.append(top_indices)
        # Record non top
        all_indices = np.arange(hidden_size)
        non_top_indices = np.setdiff1d(all_indices, top_indices)
        non_top_indices_list.append(non_top_indices)
      
    # Set all neurons not in the top list to zero
    for layer_idx in range(num_layers_modified):
        if start <= layer_idx < end:
            layer_diff = char_differences[layer_idx]
            top_indices = top_indices_list[layer_idx - start]  
            mask = np.zeros_like(layer_diff, dtype=bool)
            mask[top_indices] = True
            char_differences[layer_idx] = np.where(mask, layer_diff, 0)
        else:
            char_differences[layer_idx] = 0

    for layer_idx in range(start, end):
        layer_diff = char_differences[layer_idx]
        non_top_indices = non_top_indices_list[layer_idx - start]
        for neuron_idx in ablation_indices:
            if 0 <= neuron_idx < hidden_size:
                layer_diff[neuron_idx] = 0
                random_idx = np.random.choice(non_top_indices)
                original_val = layer_diff_original[layer_idx][random_idx]
                layer_diff[random_idx] = original_val
                char_differences[layer_idx] = layer_diff
                
    print(f"[Ablation] char_differences shape after ablation: {char_differences.shape}")
    return char_differences

def main():
    # Parse and split the arguments
    task, model_name, size, top, characters, alpha, start, end, ablation_indices = parse_arguments_and_define_characters()
    # Define paths
    # Path definition
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    json_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    matrix_path = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_mean/{model_name}"
    save_dir = os.path.join(f"/data2/paveen/RolePlaying/src/models/components/answer_modified_idx/{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    # Load difference matrices with exception handling
    try:
        data_char_diff = np.load(os.path.join(matrix_path, f'all_mean_{size}.npy'))       # (1,1,layers,hidden_size)
        data_none_char_diff = np.load(os.path.join(matrix_path, f'none_all_mean_{size}.npy')) # (1,1,layers,hidden_size)
    except FileNotFoundError as e:
        print(f"Error loading difference matrices: {e}")
        exit(1)
    
    if ablation_indices:
        print("Begain indeices ablation")
        char_differences = get_difference_matrix_ablation(
            data_char_diff,
            data_none_char_diff,
            start,
            end,
            alpha,
            top,
            ablation_indices
        )      
    else:
        print("Begin top 20 indeices ")
        top_overall = 20
        char_differences = get_difference_matrix(data_char_diff, 
                                                 data_none_char_diff, 
                                                 start, 
                                                 end, 
                                                 alpha, 
                                                 top, 
                                                 top_overall)
        
    print("char_differences shape after significant neuron masking:", char_differences.shape)
    # Initialize the model
    vc = VicundaModel(model_path=model_path)
    template = vc.template  # Assume template is a property of the model
    
    # Load the data
    data = ga.load_json_data(json_path)
    
    # Initialize accuracy counts
    accuracy_counts = {character: {"correct": 0,
                        "total": 0,
                        "E_count": 0,
                        "invalid": 0}
            for character in characters}
    
    print("Starting answer generation and accuracy calculation...")
    
    # Iterate over each sample
    for idx, sample in enumerate(data):
        context = sample.get("text", "")
        true_label_int = sample.get("label", -1)
        true_label = LABEL_MAPPING[true_label_int]
    
        for character in characters:
            # Generate the prompt
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
                generated_answer, is_correct, is_E = handle_invalid_answer(
                    vc=vc,
                    prompt=prompt,
                    true_label_text=true_label_text,
                    true_label=true_label,
                    diff_matrices=char_differences,
                    max_new_tokens=8
                )
                if is_correct:
                    ga.update_accuracy_counts(accuracy_counts, character, "correct")
                    print(f"[{idx}][{character}] '{generated_answer}' contains '{true_label_text}' -> Correct")
                elif is_E:
                    ga.update_accuracy_counts(accuracy_counts, character, "E")
                    print(f"[{idx}][{character}] '{generated_answer}' -> E")
                else:
                    ga.update_accuracy_counts(accuracy_counts, character, "invalid")
                    print(f"Sample {idx}, Character '{character}': Invalid generated answer '{generated_answer}'")
    
            # Store the generated answer
            sample[answer_key] = generated_answer
    
    # Compute accuracy
    accuracy_results = ga.compute_accuracy(accuracy_counts)
    
    # Print accuracy results
    for character, results in accuracy_results.items():
        print(f"Accuracy for {character}: {results['accuracy_percentage']}% ({results['correct']}/{results['total']})")
        print(f"Number of 'E' answers for {character}: {results['E_count']}")
        print(f"Number of invalid answers for {character}: {results['invalid']}")
    
    # Save the results to JSON
    if ablation_indices:
        save_to_json(data, accuracy_results, save_dir, task, size, top, start, end, ablation_indices)
    else:
        save_to_json(data, accuracy_results, save_dir, task, size, top, start, end)
    print("All answers and accuracy have been saved successfully.")


if __name__ == "__main__":
    main()
    