import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from vicuna import VicundaModel  # Assuming the use of the Vicuna model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inject per-sample diff via _apply_diff_hooks and extract final HS.")
    parser.add_argument("task", type=str, help="The name of the task to process.")
    parser.add_argument("size", type=str, help="Model size (e.g. '13B').")
    parser.add_argument("model", type=str, help="Model type (e.g. 'llama3').")
    parser.add_argument("--start", type=int, default=0, help="Start layer index for injecting diff (inclusive).")
    parser.add_argument("--end", type=int, default=1, help="End layer index (exclusive).")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    task = args.task
    size = args.size
    model_name = args.model

    start_layer = args.start
    end_layer = args.end

    # Load the model
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    vc = VicundaModel(model_path=model_path)  # Assuming the use of a Vicuna-based model
    template = vc.template  # Assuming this generates a prompt template
    print(f"Loaded model: {model_name}, size={size}")
    print(f"Template:\n{template}")

    # Prepare the data for processing
    mmlu_path = "/data2/paveen/RolePlaying/src/models/components/mmlu"
    json_path = os.path.join(mmlu_path, f"{task}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file '{json_path}' not found.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")

    # Load original none and expert hidden states
    hs_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3/{model_name}"
    none_hs_path = os.path.join(hs_dir, f"none_{task}_{task}_{size}.npy")
    expert_hs_path = os.path.join(hs_dir, f"{task}_{task}_{size}.npy")

    if not (os.path.isfile(none_hs_path) and os.path.isfile(expert_hs_path)):
        raise FileNotFoundError(f"Cannot find HS npy files: {none_hs_path} or {expert_hs_path}")

    none_array = np.load(none_hs_path)   # shape: (num_samples, 1, total_layers, hidden_size)
    expert_array = np.load(expert_hs_path)
    if none_array.shape != expert_array.shape:
        raise ValueError("None & Expert hidden states shape mismatch.")

    # Remove the embedding layer (index=0) and keep only the decoder layers
    none_array = none_array[:, :, 1:, :]    # shape: (num_samples, 1, num_layers, hidden_size)
    expert_array = expert_array[:, :, 1:, :]

    num_samples, _, num_layers, hidden_size = none_array.shape
    print(f"After removing embedding layer => shape: {none_array.shape}")
    print(f"  #samples={num_samples}, #layers={num_layers}, hidden_size={hidden_size}")

    # Check if the layer range is valid
    if not (0 <= start_layer < end_layer <= num_layers):
        raise ValueError(f"Invalid layer range: [start={start_layer}, end={end_layer}), "
                         f"must be in [0, {num_layers})")

    # Prepare to store final hidden states with the injected diff
    diffed_hs_list = []

    # Iterate through each sample
    print(f"Injecting diff into layers [{start_layer}:{end_layer}), extracting final HS...")
    for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
        context = sample.get("text", "")
        if not context:
            continue

        if idx >= num_samples:
            print(f"Sample idx={idx} out of range for HS array (size={num_samples}). Stop.")
            break

        # Construct the "none" prompt
        none_character = f"none {task.replace('_',' ')}"
        prompt = template.format(character=none_character, context=context)

        # Get the hidden states for the current sample
        none_hs = none_array[idx, 0]       # [num_layers, hidden_size]
        expert_hs = expert_array[idx, 0]   # Same shape

        # Calculate the diff (expert - none)
        diff_matrix = expert_hs - none_hs  # [num_layers, hidden_size]

        # Zero out diff for layers outside the [start_layer, end_layer) range
        if start_layer > 0:
            diff_matrix[:start_layer, :] = 0
        if end_layer < num_layers:
            diff_matrix[end_layer:, :] = 0

        # Call the model to inject the diff and extract the hidden states
        hs_modified_list = vc.get_hidden_states_mdf(prompt=prompt, diff_matrices=diff_matrix)

        # Usually, we only take hs_modified_list[0] (the first token's hidden state)
        if not hs_modified_list or hs_modified_list[0] is None:
            # If we don't get a correct result, skip this sample
            continue

        final_hs = hs_modified_list[0]  # shape = (num_layers, hidden_size)
        final_hs = np.expand_dims(final_hs, axis=0)  # shape => (1, num_layers, hidden_size)

        diffed_hs_list.append(final_hs)

    # Stack the diffed hidden states and save them
    if not diffed_hs_list:
        print("No diffed hidden states were collected.")
        return

    diffed_arr = np.stack(diffed_hs_list, axis=0)  # shape: (num_samples, 1, num_layers, hidden_size)

    # Prepare the directory to save the results
    save_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_diff/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save the final diffed hidden states
    out_file = os.path.join(save_dir, f"{task}_{size}_{start_layer}_{end_layer}.npy")
    np.save(out_file, diffed_arr)
    print(f"Saved diffed hidden states to: {out_file}")
    print("All done!")

if __name__ == "__main__":
    main()