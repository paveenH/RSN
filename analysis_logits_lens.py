import os
import argparse
import torch
import numpy as np
from llms import VicundaModel

def get_top_tokens_from_logits(logit_vec, tokenizer, k=5):
    """
    Input: logit_vec already in vocab-size shape.
    No matrix multiplication is needed.
    """
    values, indices = torch.topk(torch.tensor(logit_vec), k=k)
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

    # 3. Load the WHOLE Mask (Fixed range 1-33 as per your setup)
    mask_filename = f"{args.mask_type}_{args.percentage}_1_33_{args.size}.npy"
    mask_path = os.path.join(MASK_DIR, mask_filename)

    if not os.path.exists(mask_path):
        print(f"[ERROR] Mask file not found: {mask_path}")
        return

    print(f"[INFO] Loading Whole Mask: {mask_path}")
    # Mask shape: [Total_Layers, Hidden_Dim] (e.g., [32, 4096] depending on model)
    full_mask = np.load(mask_path)
    total_layers_in_mask = full_mask.shape[0]
    print(f"[INFO] Mask Shape: {full_mask.shape}")

    # 4. Determine Layer Range to Analyze
    # If args.analyze_layers is provided (e.g., "11-20"), analyze that slice.
    # Otherwise, analyze the whole mask (1 to total_layers).
    if args.analyze_layers:
        start_layer, end_layer = map(int, args.analyze_layers.split("-"))
        print(f"\n>>> Analyzing Specified Layer Range: [{start_layer}, {end_layer})")
    else:
        start_layer, end_layer = 1, total_layers_in_mask + 1
        print(f"\n>>> Analyzing WHOLE Mask Range: [{start_layer}, {end_layer})")

    # Handle indexing: Layer 1 usually corresponds to index 0 in the mask if mask excludes embedding
    # Assuming mask index 0 -> Layer 1
    # Slice range for array indexing
    slice_start = start_layer - 1
    slice_end = end_layer - 1
    
    # Safety check
    slice_start = max(0, slice_start)
    slice_end = min(total_layers_in_mask, slice_end)

    # =====================================================
    # Mode A — Top-K Strongest Neurons in Range
    # =====================================================
    if args.trace_index is None:
        print(f"\n>>> [Mode A] Finding Top-{args.topk} Strongest Active Neurons in Layers {start_layer}-{end_layer}")
        
        # Extract the slice of interest
        mask_slice = full_mask[slice_start:slice_end, :]
        
        # Get indices of ALL non-zero elements in this slice
        # Since it's a sparse mask, this list is short (e.g., 20 * num_layers)
        # We want to sort them by magnitude (absolute value)
        flat_indices = np.argsort(-np.abs(mask_slice).flatten())[:args.topk]
        
        print(f"{'Layer':<6} | {'Index':<6} | {'Val (Δμ)':<10} | {'Logit Lens (Top Tokens)'}")
        print("-" * 90)

        for flat_idx in flat_indices:
            # Convert flat index back to relative (row, col) in the slice
            rel_layer, neuron_idx = divmod(flat_idx, hidden_dim)
            
            # Map back to absolute layer number
            # abs_layer_idx is the index in full_mask
            abs_layer_idx = slice_start + rel_layer
            # display_layer is the human-readable layer number (1-based)
            display_layer = abs_layer_idx + 1 
            
            val = full_mask[abs_layer_idx, neuron_idx]
            
            # Skip if value is effectively zero (just in case)
            if abs(val) < 1e-9: continue

            # Logit Lens Calculation
            neuron_vec = lm_head[:, neuron_idx]
            projected_vec = neuron_vec * val  # Signed projection
            
            top_tokens = get_top_tokens_from_logits(projected_vec, vc.tokenizer)
            
            print(f"{display_layer:<6} | {neuron_idx:<6} | {val:<10.4f} | {top_tokens}")

    # =====================================================
    # Mode B — Trace Specific Index
    # =====================================================
    else:
        idx = args.trace_index
        print(f"\n>>> [Mode B] Tracing Neuron Index {idx} across Layers {start_layer}-{end_layer}")
        print(f"{'Layer':<6} | {'Val (Δμ)':<10} | {'Logit Lens Analysis'}")
        print("-" * 90)

        for layer_idx in range(slice_start, slice_end):
            val = full_mask[layer_idx, idx]
            display_layer = layer_idx + 1
            
            # Only show layers where this neuron is actually selected (non-zero)
            # OR show all if you want to see continuity even if zeroed out
            # Here we show if non-zero to reduce noise, or just trace it anyway.
            # Let's show all to see the "vertical striation" pattern (including gaps).
            
            neuron_vec = lm_head[:, idx]
            
            # Static: What does this neuron mean naturally?
            static_tokens = get_top_tokens(neuron_vec, lm_head, vc.tokenizer, k=3)
            
            # Dynamic: What is it doing in this role?
            # If val is 0, dynamic is all 0s (meaningless), so handle that
            if abs(val) > 1e-9:
                dynamic_vec = neuron_vec * val
                dynamic_tokens = get_top_tokens(dynamic_vec, lm_head, vc.tokenizer, k=3)
                status_str = f"Dynamic: [{dynamic_tokens}]"
            else:
                status_str = "(Inactive in Mask)"

            print(f"{display_layer:<6} | {val:<10.4f} | Static: [{static_tokens}] -> {status_str}")

    print("\nAll tasks finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    
    # NOTE: configs arg is removed because we are loading a fixed 1-33 mask file directly.
    # Added specific layer range argument for analysis
    parser.add_argument("--analyze_layers", type=str, default=None, 
                        help="Specific layer range to analyze, e.g., '11-20'. If None, analyzes whole mask.")

    parser.add_argument("--mask_type", type=str, default="nmd", help="nmd, ttest_layer, selection_pca")
    parser.add_argument("--data", type=str, default="data2")

    # Analysis options
    parser.add_argument("--topk", type=int, default=20, help="Top-K neurons to show")
    parser.add_argument("--trace_index", type=int, default=None, help="Index to trace")

    args = parser.parse_args()

    # Path setup
    MASK_DIR = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    
    main()