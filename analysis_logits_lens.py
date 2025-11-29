import os
import argparse
import torch
import numpy as np
import csv
from llms import VicundaModel

def get_top_tokens_from_logits(logit_vec, tokenizer, k=5):
    """
    Input: logit_vec already in vocab-size shape.
    """
    # Ensure it's a tensor
    if not isinstance(logit_vec, torch.Tensor):
        logit_vec = torch.tensor(logit_vec)
        
    values, indices = torch.topk(logit_vec, k=k)
    tokens = []
    for v, i in zip(values, indices):
        token_str = tokenizer.decode([i.item()]).replace('\n', '\\n')
        tokens.append(f"'{token_str}'({v:.2f})")
    return ", ".join(tokens)

def main():
    # 1. Load Model
    print(f"[INFO] Loading Model from {args.model_dir} ...")
    vc = VicundaModel(model_path=args.model_dir)
    
    # 2. Extract Unembedding Matrix
    lm_head = vc.model.lm_head.weight.detach().float().cpu()
    vocab_size, hidden_dim = lm_head.shape
    print(f"[INFO] Unembedding Matrix Shape: {lm_head.shape}")

    # 3. Load Mask
    mask_filename = f"{args.mask_type}_{args.percentage}_1_33_{args.size}.npy"
    mask_path = os.path.join(MASK_DIR, mask_filename)

    if not os.path.exists(mask_path):
        print(f"[ERROR] Mask file not found: {mask_path}")
        return

    print(f"[INFO] Loading Mask: {mask_path}")
    full_mask = np.load(mask_path) # [32, 4096]
    total_layers_in_mask = full_mask.shape[0]

    # 4. Determine Layer Range
    if args.analyze_layers:
        start_layer, end_layer = map(int, args.analyze_layers.split("-"))
    else:
        start_layer, end_layer = 1, total_layers_in_mask + 1
    
    slice_start = start_layer - 1
    slice_end = end_layer - 1
    slice_start = max(0, slice_start)
    slice_end = min(total_layers_in_mask, slice_end)

    # 5. Prepare Output CSV
    csv_file = "analysis_results.csv"
    csv_columns = ["Index", "Sign", "Layers", "Static_Tokens", "Dynamic_Tokens", "Max_Val", "Full_Log"]
    results_data = []

    print(f"\n{'='*20} BATCH TRACE ANALYSIS {'='*20}")
    print(f"Target Indices: {args.trace_indices}")
    print(f"{'Index':<6} | {'Sign':<4} | {'Layers':<20} | {'Logit Lens Analysis'}")
    print("-" * 120)

    # 6. Iterate through requested indices
    for idx in args.trace_indices:
        if idx >= hidden_dim:
            print(f"[WARN] Index {idx} out of bounds, skipping.")
            continue

        # Storage for aggregation
        pos_layers = []
        neg_layers = []
        
        # We need to pick a representative value to calculate Dynamic tokens.
        # Strategy: Pick the value with the MAX ABSOLUTE magnitude in that group.
        max_pos_val = -1.0
        max_neg_val = 1.0 
        
        # 6.1 Scan layers to group by sign
        for layer_idx in range(slice_start, slice_end):
            val = full_mask[layer_idx, idx]
            display_layer = layer_idx + 1
            
            if val > 1e-5: # Positive
                pos_layers.append(display_layer)
                if val > max_pos_val: max_pos_val = val
            elif val < -1e-5: # Negative
                neg_layers.append(display_layer)
                if val < max_neg_val: max_neg_val = val
        
        # 6.2 Get Static Vectors (Constant for this index)
        neuron_vec = lm_head[:, idx]
        static_tokens = get_top_tokens_from_logits(neuron_vec, vc.tokenizer, k=5)

        # 6.3 Generate Outputs for Positive Group
        if pos_layers:
            # Calculate Dynamic using the strongest positive activation
            dynamic_vec = neuron_vec * max_pos_val
            dynamic_tokens = get_top_tokens_from_logits(dynamic_vec, vc.tokenizer, k=5)
            
            layer_str = ",".join(map(str, pos_layers))
            full_log = f"Static: [{static_tokens}] -> Dynamic: [{dynamic_tokens}]"
            
            print(f"{idx:<6} | {'+':<4} | {layer_str:<20} | {full_log}")
            
            results_data.append({
                "Index": idx,
                "Sign": "+",
                "Layers": layer_str,
                "Static_Tokens": static_tokens,
                "Dynamic_Tokens": dynamic_tokens,
                "Max_Val": f"{max_pos_val:.4f}",
                "Full_Log": full_log
            })

        # 6.4 Generate Outputs for Negative Group
        if neg_layers:
            # Calculate Dynamic using the strongest negative activation
            dynamic_vec = neuron_vec * max_neg_val
            dynamic_tokens = get_top_tokens_from_logits(dynamic_vec, vc.tokenizer, k=5)
            
            layer_str = ",".join(map(str, neg_layers))
            full_log = f"Static: [{static_tokens}] -> Dynamic: [{dynamic_tokens}]"
            
            print(f"{idx:<6} | {'-':<4} | {layer_str:<20} | {full_log}")

            results_data.append({
                "Index": idx,
                "Sign": "-",
                "Layers": layer_str,
                "Static_Tokens": static_tokens,
                "Dynamic_Tokens": dynamic_tokens,
                "Max_Val": f"{max_neg_val:.4f}",
                "Full_Log": full_log
            })

    # 7. Save to CSV
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, escapechar='\\')
            
            writer.writeheader()
            for data in results_data:
                writer.writerow(data)
        print(f"\n[SUCCESS] Results saved to {csv_file}")
    except IOError as e:
        print(f"[ERROR] Could not write to CSV file: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error while writing CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--analyze_layers", type=str, default=None)
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--data", type=str, default="data2")
    
    # Changed: trace_indices accepts a list
    parser.add_argument("--trace_indices", type=int, nargs='+', 
                        default=[2629, 2692, 1731, 4055,  373, 3585, 3070,  133,  873, 1298, 2646,
                                 1421, 2352, 1189,  291, 3695, 3516, 2932, 2184, 2265,  761, 2082,
                                 384, 1130, 2977, 1122, 2303, 3266,  281],
                        help="List of neuron indices to trace")
    
    # Mode A topk (optional, keep for compatibility but main logic uses trace_indices)
    parser.add_argument("--topk", type=int, default=20)

    args = parser.parse_args()

    # Path setup
    MASK_DIR = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    
    main()