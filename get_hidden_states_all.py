#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch hidden-state extraction for multiple tasks, sizes, and models.
Author: paveenhuang
"""

import os
import json
import numpy as np
from tqdm import tqdm
from llms import VicundaModel
from get_answer_alltasks import make_characters, TASKS

# ── Configuration ────────────────────────────────────────────────────────────
MODEL = "llama3"
SIZE = "3B"
TYPE = "non"

# MODEL_DIR = os.path.join("/data2/paveen/RolePlaying/shared", model, size)
# MODEL_DIR = "openchat/openchat_3.5"
# MODEL_DIR = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_DIR = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_DIR = "HuggingFaceH4/zephyr-7b-beta"
# MODEL_DIR = "NousResearch/Hermes-3-Llama-3.2-3B"

print(MODEL_DIR)

DIFFUSION = None 
# Output base directory for hidden states
BASE_SAVE_DIR = f"/data2/paveen/RolePlaying/components/hidden_states_{TYPE}"
# Path to MMLU JSON files
PATH_MMLU = "/data2/paveen/RolePlaying/components/mmlu"
        

print(f"Loading model {MODEL}/{SIZE}...")
vc = VicundaModel(model_path=MODEL_DIR)
template = vc.template
print(f"Template: {template}\n")

model_save_dir = os.path.join(BASE_SAVE_DIR, MODEL)
os.makedirs(model_save_dir, exist_ok=True)

for task in TASKS:
    # Load JSON
    json_path = os.path.join(PATH_MMLU, f"{task}.json")
    print(f"Loading JSON data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total samples for {task}: {len(data)}")

    # Storage for hidden states
    characters = make_characters(task, TYPE)
    print("characters:", characters)
    hidden_states_storage = {ch: [] for ch in characters}

    # Extract hidden states per sample
    print(f"Processing task {task}...")
    for idx, sample in enumerate(tqdm(data, desc=f"{MODEL}/{SIZE}/{task}")):
        context = sample.get("text", "")
        if not context:
            tqdm.write(f"Sample {idx} missing text; skipping.")
            continue

        for character in characters:
            prompt = template.format(character=character, context=context)
            hidden_states = vc.get_hidden_states(prompt=prompt, character=character)
            # Validate
            if any(pos is None for pos in hidden_states):
                tqdm.write(f"Sample {idx}, '{character}' missing hidden states; skipping.")
                continue
            # Stack and store
            hs_array = np.stack([np.stack(pos, axis=0) for pos in hidden_states], axis=0)
            hidden_states_storage[character].append(hs_array)

    # Save hidden states
    print(f"Saving hidden states for {task}...")
    for character, hs_list in hidden_states_storage.items():
        if not hs_list:
            tqdm.write(f"No hidden states for '{character}' in task {task}; skipping save.")
            continue
        hs_array = np.stack(hs_list, axis=0)
        char_safe = character.replace(" ", "_")
        char_safe = char_safe.replace("-", "_")
        save_path = os.path.join(model_save_dir, f"{char_safe}_{task}_{SIZE}.npy")
        np.save(save_path, hs_array)
        print(f"Saved: {save_path}")
print("All models and tasks processed.")
