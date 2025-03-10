import os
import numpy as np
import json
import argparse

# Define the tasks to process
TASKS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_medicine",
    "college_mathematics", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous",
    "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and save mean HS-diff for each answer pair.")
    parser.add_argument("model", type=str, help="Name of the model (e.g., llama3)")
    parser.add_argument("size", type=str, help="Size of the model (e.g., 1B)")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    model = args.model
    size = args.size

    # Define paths for data loading and saving results
    current_path = os.getcwd()
    hidden_states_path = os.path.join(current_path, "hidden_states_v3", model)
    save_path = os.path.join(current_path, "hidden_states_v3_pair", model)
    os.makedirs(save_path, exist_ok=True)
    json_path = os.path.join(current_path, "answer", f"{model}_v3")
    
    # Define valid answer set (A-E)
    valid_answers = set(['A', 'B', 'C', 'D', 'E'])

    # Initialize pair data dictionary
    pair_data = {}
    for a in valid_answers:
        for b in valid_answers:
            pair_key = f"{a}-{b}"
            pair_data[pair_key] = []

    # Iterate over tasks
    for task in TASKS:
        print(f"Processing task: {task}")

        # Define file paths for the hidden states and JSON answers
        data_char_filepath = os.path.join(hidden_states_path, f"{task}_{task}_{size}.npy")
        data_none_char_filepath = os.path.join(hidden_states_path, f"none_{task}_{task}_{size}.npy")
        json_filepath = os.path.join(json_path, f"{task}_{size}_answers.json")
        
        # Check if files exist
        if not os.path.exists(data_char_filepath):
            print(f"Data char file not found: {data_char_filepath}")
            continue
        if not os.path.exists(data_none_char_filepath):
            print(f"Data none-char file not found: {data_none_char_filepath}")
            continue
        if not os.path.exists(json_filepath):
            print(f"JSON file not found: {json_filepath}")
            continue
        
        # Load hidden states data
        data_char = np.load(data_char_filepath)
        data_none_char = np.load(data_none_char_filepath)
        
        # Load JSON data
        with open(json_filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        samples = json_data.get("data", [])
        
        # Check if there are any samples
        if len(samples) == 0:
            print(f"No samples found for task: {task}")
            continue
        
        # Check sample count consistency
        if data_char.shape[0] != len(samples) or data_none_char.shape[0] != len(samples):
            print(f"Mismatch in sample count for task: {task}")
            continue
        
        # Calculate diff for each sample
        for i, entry in enumerate(samples):
            ans_none = entry.get(f"answer_none_{task}")
            ans_expert = entry.get(f"answer_{task}")
            if ans_none is None or ans_expert is None:
                continue
            if (ans_none not in valid_answers) or (ans_expert not in valid_answers):
                continue

            pair_key = f"{ans_none}-{ans_expert}"
            expert_hs = np.squeeze(data_char[i])
            none_hs = np.squeeze(data_none_char[i])
            diff = expert_hs - none_hs
            
            pair_data[pair_key].append(diff)
    
    # Compute mean diff for each pair
    mean_pair_data = {}
    for pair, diffs in pair_data.items():
        if diffs:
            diffs_stack = np.stack(diffs, axis=0)
            mean_diff = diffs_stack.mean(axis=0, keepdims=True)
            mean_pair_data[pair] = mean_diff
            print(f"Pair {pair}: {len(diffs)} samples, mean shape: {mean_diff.shape}")
        else:
            print(f"Pair {pair}: No samples collected.")
    
    # Save the results
    out_file = os.path.join(save_path, f"all_pairs_mean_{size}.npz")
    np.savez(out_file, **mean_pair_data)
    print(f"Saved mean difference for each answer pair to {out_file}")

if __name__ == "__main__":
    main()