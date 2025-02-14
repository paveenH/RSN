#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:36:29 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:00:33 2025

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

    # Split the combined argument into task, model, size, top, alpha, start, and end
    try:
        task, model, size, top, alpha, start, end = args.task_size.split()
    except ValueError:
        raise ValueError("The task_size parameter should contain seven parts: task, model, size, top, alpha, start, and end, separated by spaces.")

    # Define characters based on the task
    task_name = task.replace('_', ' ')
    characters = [f"none {task_name}", task_name]

    return task, model, size, int(top), characters, float(alpha), int(start), int(end)


def regenerate_answer(vc, prompt, model, char_differences):
    """
    Generate an answer using VicundaModel, cleaning the output based on the model type.
    """
    if model.lower() == "phi":
        generated_output = vc.regenerate(
            [prompt],
            diff_matrices=char_differences, 
            max_new_tokens=6
        )[0]
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
    generated_output_long = vc.regenerate(
        [prompt],
        diff_matrices=diff_matrices,
        max_new_tokens=max_new_tokens
    )[0]
    generated_answer = generated_output_long.strip()
    
    # Apply cleaning to extract a potential valid answer
    extracted_answer = ga.cleaning(generated_answer)
    
    # Check if the extracted answer is valid
    if extracted_answer in ["A", "B", "C", "D"] and extracted_answer == true_label:
        return "[Add]" + extracted_answer + " original:" + generated_answer, True
    
    # Fallback: Check if the correct answer text is contained in the generated output
    elif true_label_text and true_label_text.lower() in generated_answer.lower():
        return "[Add]" + generated_answer, True
    
    # If no valid answer is found, return the output as invalid
    return generated_answer, False


def save_to_json(data, accuracy_results, save_dir, task, size, top, alpha):
    """
    Save the generated answers and accuracy to a JSON file.
    """
    final_output = {
        "data": data,
        "accuracy": accuracy_results,
    }
    answers_save_path = os.path.join(save_dir, f"{task}_{size}_answers_{top}.json")
    print("Saving generated answers and accuracy to JSON...")
    with open(answers_save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"Saved answers and accuracy to {answers_save_path}")


def main():
    # Parse and split the arguments
    task, model_name, size, top, characters, alpha, start, end = parse_arguments_and_define_characters()
    # Define paths
    # Path definition
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    json_path = os.path.join("/data2/paveen/RolePlaying/src/models/components/mmlu", f"{task}.json")
    matrix_path = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_mean/{model_name}"
    save_dir = os.path.join(f"/data2/paveen/RolePlaying/src/models/components/answer_modified/{model_name}/{alpha}")
    os.makedirs(save_dir, exist_ok=True)

    # Load difference matrices with exception handling
    try:
        data_char_diff = np.load(os.path.join(matrix_path, f'all_mean_{size}.npy'))       # (1,1,layers,hidden_size)
        data_none_char_diff = np.load(os.path.join(matrix_path, f'none_all_mean_{size}.npy')) # (1,1,layers,hidden_size)
    except FileNotFoundError as e:
        print(f"Error loading difference matrices: {e}")
        exit(1)
    
    # Compute the difference matrix and apply scaling with alpha
    char_differences = (data_char_diff - data_none_char_diff).squeeze(0).squeeze(0)    # (layers,hidden_size)
    
    # Ensure start and end are in range
    num_layers = char_differences.shape[0]
    start = max(0, min(start, num_layers - 1))
    end = max(start + 1, min(end, num_layers))
    
    char_differences = char_differences[start:end] * alpha
    
    # Debug
    print(f"data_char_diff shape: {data_char_diff.shape}")
    print(f"data_none_char_diff shape: {data_none_char_diff.shape}")
    print(f"char_differences shape: {char_differences.shape}")
    
    if top >= 0:
        print(f"Top {top} calculation begin.")
        for layer_idx in range(char_differences.shape[0]): 
            layer_diff = char_differences[layer_idx]  # (hidden_size,)
            top_indices = np.argsort(np.abs(layer_diff))[-top:]   # Top N
            mask = np.zeros_like(layer_diff, dtype=bool)
            mask[top_indices] = True
            char_differences[layer_idx] = np.where(mask, layer_diff, 0)
    
    # Debug
    print(f"char_differences shape after top-{top} masking: {char_differences.shape}")
    
    # Initialize the model
    vc = VicundaModel(model_path=model_path)
    template = vc.template  # Assume template is a property of the model
    
    # Load the data
    data = ga.load_json_data(json_path)
    
    # Initialize accuracy counts
    accuracy_counts = {character: {"correct": 0, "total": 0, "E_count": 0, "invalid": 0} for character in characters}
    
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
                generated_answer, is_correct = handle_invalid_answer(
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
    save_to_json(data, accuracy_results, save_dir, task, size, top, alpha)
    
    print("All answers and accuracy have been saved successfully.")


if __name__ == "__main__":
    main()