import os
import argparse
import torch
import numpy as np
import csv
from llms import VicundaModel  

def format_token_output(values, indices, tokenizer):
    """
    Helper function to format token strings with their logit contributions.
    """
    tokens = []
    for v, i in zip(values, indices):
        # Decode token and escape newlines for CSV safety
        token_str = tokenizer.decode([i.item()]).replace('\n', '\\n').replace('"', '""')
        tokens.append(f"'{token_str}'({v:.2f})")
    return ", ".join(tokens)

def main():
    # ==========================================
    # 1. Setup & Load Model
    # ==========================================
    print(f"[INFO] Loading Model from {args.model_dir} ...")
    try:
        vc = VicundaModel(model_path=args.model_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Extract Unembedding Matrix (Output Head)
    # Shape: [vocab_size, hidden_dim]
    # Move to GPU for faster matrix multiplication if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    lm_head = vc.model.lm_head.weight.detach().to(device)
    vocab_size, hidden_dim = lm_head.shape
    print(f"[INFO] Unembedding Matrix Shape: {lm_head.shape}")

    # ==========================================
    # 2. Load Direction / Mask File
    # ==========================================
    # This file is expected to be [Layers, Hidden_Dim]
    # It represents the Δμ (Expert - NonExpert) or any other direction vector
    mask_filename = f"{args.mask_type}_{args.percentage}_1_33_{args.size}.npy"
    mask_path = os.path.join(MASK_DIR, mask_filename)

    if not os.path.exists(mask_path):
        print(f"[ERROR] Direction file not found: {mask_path}")
        return

    print(f"[INFO] Loading Direction Vector: {mask_path}")
    # Load and move to same device as lm_head
    direction_matrix = np.load(mask_path) 
    direction_matrix = torch.tensor(direction_matrix, dtype=torch.float32).to(device)
    
    total_layers_in_mask = direction_matrix.shape[0]
    print(f"[INFO] Direction Matrix Shape: {direction_matrix.shape}")

    # ==========================================
    # 3. Prepare Analysis Output
    # ==========================================
    csv_file = "direction_logit_lens_results.csv"
    csv_columns = [
        "Layer_Index", 
        "Interpretation", 
        "Top_Positive_Tokens (Expert Adds)", 
        "Top_Negative_Tokens (Expert Removes/Non-Exp Adds)"
    ]
    results_data = []

    print(f"\n{'='*30} DIRECTION-LEVEL LOGIT LENS {'='*30}")
    print(f"{'Layer':<6} | {'Top Positive Concepts':<50} | {'Top Negative Concepts'}")
    print("-" * 130)

    # ==========================================
    # 4. Layer-wise Direction Analysis
    # ==========================================
    # Iterate through all layers in the mask
    for layer_idx in range(total_layers_in_mask):
        
        # --- A. Get the Direction Vector for this Layer ---
        # Shape: [hidden_dim]
        # This vector represents the aggregate "Expertness" direction for this layer
        layer_vec = direction_matrix[layer_idx]

        # Check if the vector is all zeros (dead layer)
        if torch.all(layer_vec == 0):
            continue

        # --- B. Project to Vocabulary Space ---
        # Mathematical operation: Logits = Vector @ Unembedding_Matrix.T
        # Shape: [hidden] @ [hidden, vocab] -> [vocab]
        # or in torch: [vocab, hidden] x [hidden] -> [vocab]
        projected_logits = torch.matmul(lm_head, layer_vec)

        # --- C. Analyze Positive End (Expert Enforces) ---
        # What concepts does this direction maximally Activate?
        top_vals, top_inds = torch.topk(projected_logits, k=args.topk)
        pos_tokens_str = format_token_output(top_vals, top_inds, vc.tokenizer)

        # --- D. Analyze Negative End (Expert Suppresses / Non-Expert Manifests) ---
        # What concepts does this direction maximally Suppress?
        # distinct from "not activating", these are the most negative logits
        bot_vals, bot_inds = torch.topk(projected_logits, k=args.topk, largest=False)
        bot_tokens_str = format_token_output(bot_vals, bot_inds, vc.tokenizer)

        # --- E. Simple Heuristic Interpretation (Optional) ---
        # Just a quick tag to help reading the CSV
        interpretation = "Neutral"
        if "..." in bot_tokens_str or "sure" in bot_tokens_str:
            interpretation = "Hesitation Suppression"
        elif any(x in pos_tokens_str for x in ["{", "}", "_", "#"]):
            interpretation = "Structural/Code"

        # --- F. Print & Store ---
        print(f"{layer_idx+1:<6} | {pos_tokens_str[:48]:<50} | {bot_tokens_str[:50]}")

        results_data.append({
            "Layer_Index": layer_idx + 1,
            "Interpretation": interpretation,
            "Top_Positive_Tokens (Expert Adds)": pos_tokens_str,
            "Top_Negative_Tokens (Expert Removes/Non-Exp Adds)": bot_tokens_str
        })

    # ==========================================
    # 5. Save Results
    # ==========================================
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, escapechar='\\')
            writer.writeheader()
            for data in results_data:
                writer.writerow(data)
        print(f"\n[SUCCESS] Analysis complete. Saved to {csv_file}")
    except Exception as e:
        print(f"[ERROR] Could not write CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction-level Logit Lens Analysis")

    # Reuse your existing arguments for compatibility
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs", type=str, default="llama3")
    parser.add_argument("--size", type=str, default="8B")
    parser.add_argument("--type", type=str, default="non") # expert vs non, etc.
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--mask_type", type=str, default="nmd") # e.g., 'nmd' for mean difference
    parser.add_argument("--data", type=str, default="data2")
    
    # Analysis params
    parser.add_argument("--topk", type=int, default=10, help="Number of top tokens to show")

    args = parser.parse_args()

    # Path Construction (Adjust as per your file structure)
    # Assuming the same structure as your previous snippet
    MASK_DIR = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    
    main()