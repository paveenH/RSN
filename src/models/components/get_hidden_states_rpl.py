#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 17:08:11 2025

@author: paveenhuang

Example script to replace the hidden states (in range [start, end)) 
of 'None Expert' with 'Expert' hidden states for each sample, 
and observe the resulting model output or hidden states.
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from vicuna import VicundaModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Replace None hidden states with Expert states in specified layer range.")
    parser.add_argument("task", type=str, help="The name of the task to process.")
    parser.add_argument("size", type=str, help="Model size (e.g. '13B').")
    parser.add_argument("model", type=str, help="Model type (e.g. 'llama3').")
    parser.add_argument("--start", type=int, default=0, help="Start layer index for replacement (inclusive).")
    parser.add_argument("--end", type=int, default=1, help="End layer index for replacement (exclusive).")
    parser.add_argument("--max_new_tokens", type=int, default=1, help="How many new tokens to generate.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Sampling top_p.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--save_output", action="store_true", help="Whether to save final model outputs to disk.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    task = args.task
    size = args.size
    model_name = args.model

    start_layer = args.start
    end_layer = args.end

    # -----------------------------
    # 1) Load model & prompt
    # -----------------------------
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    vc = VicundaModel(model_path=model_path)
    template = vc.template
    print(f"Loaded model: {model_name}, size={size}, template:\n{template}")

    # -----------------------------
    # 2) Prepare data & path
    # -----------------------------
    # Json path
    mmlu_path = "/data2/paveen/RolePlaying/src/models/components/mmlu"
    json_path = os.path.join(mmlu_path, f"{task}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file '{json_path}' not found.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")

    # hidden_states path
    hs_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v5/{model_name}"
    none_hs_path = os.path.join(hs_dir, f"none_{task}_{size}.npy")  # none Expert
    expert_hs_path = os.path.join(hs_dir, f"{task}_{size}.npy")     # Expert

    if not (os.path.isfile(none_hs_path) and os.path.isfile(expert_hs_path)):
        raise FileNotFoundError(f"Cannot find HS npy files: {none_hs_path} or {expert_hs_path}")

    # Load: shape (num_samples, 1, 33, hidden_size)
    none_array = np.load(none_hs_path)
    expert_array = np.load(expert_hs_path)
    if none_array.shape != expert_array.shape:
        raise ValueError("None & Expert hidden states shape mismatch.")

    # Cut off layer 0 (embedding) and keep the transformer layer (1..32)
    # (num_samples, 1, 32, hidden_size)
    none_array = none_array[:, :, 1:, :]
    expert_array = expert_array[:, :, 1:, :]

    num_samples, _, num_layers, hidden_size = none_array.shape
    print(f"After removing embedding layer, shape => {none_array.shape}")
    print(f"  #samples={num_samples}, #layers={num_layers}, hidden_size={hidden_size}")

    # Check start/end
    if not (0 <= start_layer < end_layer <= num_layers):
        raise ValueError(
            f"Invalid layer range [start={start_layer}, end={end_layer}). "
            f"Must be within [0, {num_layers}]"
        )

    # Optional: to sace output
    all_outputs = []

    # -----------------------------
    # 3) Replace and infer samples one by one
    # -----------------------------
    print(f"Replacing None hidden states with Expert in layers [{start_layer}:{end_layer}), then generate...")
    for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
        context = sample.get("text", "")
        if not context:
            continue

        # COnstruct none Prompt
        none_character = f"none {task.replace('_',' ')}"
        prompt = template.format(character=none_character, context=context)

        # If idx exceeds the number of hs, it will not be processed
        if idx >= num_samples:
            print(f"Index {idx} out of range for hidden states array (size={num_samples}). Stopping loop.")
            break

        # (num_layers, hidden_size)
        none_hs = none_array[idx, 0]   
        expert_hs = expert_array[idx, 0]

        # Assemble replace_matrices: size (num_layers, hidden_size)
        replace_matrices = none_hs.copy()  # The default is the same as none
        # Replace the interval [start_layer, end_layer) with expert
        replace_matrices[start_layer:end_layer] = expert_hs[start_layer:end_layer]

        # Call replace_generate
        generated_texts = vc.replace_generate(
            inputs=[prompt],
            replace_matrices=replace_matrices,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            start=start_layer,      
            end=end_layer,
        )
        gen_answer = generated_texts[0] if generated_texts else ""

        # Optionally save output
        if args.save_output:
            all_outputs.append({
                "index": idx,
                "none_prompt": prompt,
                "gen_answer": gen_answer
            })

    # -----------------------------
    # 4) Optional: Store the final result
    # -----------------------------
    if args.save_output:
        output_dir = "./replace_outputs"
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"replace_{task}_{size}_{start_layer}_{end_layer}.json")
        with open(out_file, "w", encoding="utf-8") as fout:
            json.dump(all_outputs, fout, indent=2, ensure_ascii=False)
        print(f"\nSaved replaced-generation results to: {out_file}")


if __name__ == "__main__":
    main()