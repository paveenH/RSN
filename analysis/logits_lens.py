import os
import argparse
import torch
import numpy as np
import utils
from vicunda_wrapper import VicundaModel  # assuming this is your wrapper


def get_top_tokens(vector, unembed_matrix, tokenizer, k=5):
    """
    Core of Logit-Lens:
    Compute logits = vector @ Unembedding_Matrix.T

    vector: [Hidden_Dim]
    unembed_matrix: [Vocab_Size, Hidden_Dim]
    """
    # Project into vocabulary logit space
    # logits shape = [Vocab_Size]
    logits = torch.matmul(unembed_matrix, vector)

    # Top-K selection
    values, indices = torch.topk(logits, k=k)

    tokens = []
    for v, i in zip(values, indices):
        token_str = tokenizer.decode([i.item()]).replace('\n', '\\n')
        tokens.append(f"'{token_str}'({v:.2f})")

    return ", ".join(tokens)


def main():
    # 1. Parse alpha/layer-range configs
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    # 2. Load model
    print(f"[INFO] Loading Model from {args.model_dir} ...")
    vc = VicundaModel(model_path=args.model_dir)

    # 3. Extract Unembedding Matrix (LM head)
    # LLaMA/Qwen lm_head weight shape is typically [Vocab, Hidden]
    lm_head = vc.model.lm_head.weight.detach().float().cpu()
    vocab_size, hidden_dim = lm_head.shape
    print(f"[INFO] Unembedding Matrix Shape: {lm_head.shape}")

    # 4. Run analysis for each (alpha, start_layer, end_layer) config
    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        print(f"\n{'='*20} Config: Alpha={alpha}, Layers={st}-{en} {'='*20}")

        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)

        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found: {mask_path}")
            continue

        print(f"[INFO] Loading Mask: {mask_path}")
        # Mask shape = [Layers, Hidden_Dim] (e.g., [33, 4096])
        mask = np.load(mask_path)
        
        # The mask is sparse already, only selected neruons in layer range have value, others is zeros

        # =====================================================
        # Mode A — Global Top-K strongest neurons (Hub Neurons)
        # =====================================================
        # Analuze the selected neurons (which have non-zero value 1) layer by layer 2) whole mask.)
        print(f"\n>>> [Mode A] Analyzing Top-{args.topk} Strongest Neurons in Range [{st}, {en})")

        
        # Find top-K absolute values across the slice
        # DO not have to find neurons again, just get the index with non-zero neuros 
        # flat_indices = np.argsort(-np.abs(mask_slice).flatten())[:args.topk]
        
        # Find the indeice layer by layer 
        for flat_idx in flat_indices:
            # # Convert back to (layer, neuron_idx)
            # rel_layer, neuron_idx = divmod(flat_idx, hidden_dim)
            # abs_layer = st + rel_layer

            # Raw Δμ value
            val = mask[abs_layer, neuron_idx]

            # Unembedding vector for this neuron: lm_head[:, neuron_idx]
            neuron_vec = lm_head[:, neuron_idx]

            # Apply signed strength — positive increases probability, negative decreases
            projected_vec = neuron_vec * val

            top_tokens = get_top_tokens(projected_vec, lm_head, vc.tokenizer)

            print(f"{abs_layer:<6} | {neuron_idx:<6} | {val:<10.4f} | {top_tokens}")

        # =====================================================
        # Mode B — Vertical Striation Trace
        # Track a single neuron index across layers
        # =====================================================
        if args.trace_index is not None:
            idx = args.trace_index
            print(f"\n>>> [Mode B] Tracing Vertical Striation for Neuron Index {idx}")
            print(f"{'Layer':<6} | {'Val (Δμ)':<10} | {'Logit Lens'}")
            print("-" * 90)

            for layer in range(st, en):
                val = mask[layer, idx]

                neuron_vec = lm_head[:, idx]

                # --- Static Meaning ---
                # Top tokens for the raw channel (not multiplied by val)
                static_tokens = get_top_tokens(neuron_vec, lm_head, vc.tokenizer, k=3)

                # --- Dynamic Action ---
                dynamic_vec = neuron_vec * val
                dynamic_tokens = get_top_tokens(dynamic_vec, lm_head, vc.tokenizer, k=3)

                print(f"{layer:<6} | {val:<10.4f} | Static: [{static_tokens}]  ->  Dynamic: [{dynamic_tokens}]")

    print("\nAll tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logit-Lens Analysis for RSNs.")

    # Original parameters unchanged
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"],
                        help="List of alpha-start-end triplets")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type (nmd/random)")
    parser.add_argument("--ans_file", type=str, default="answer_mdf")
    parser.add_argument("--E", dest="use_E", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--tail_len", type=int, default=1)
    parser.add_argument("--suite", type=str, default="default")
    parser.add_argument("--data", type=str, default="data1")

    # --- New analysis options ---
    parser.add_argument("--topk", type=int, default=10,
                        help="Display top-K strongest neurons")
    parser.add_argument("--trace_index", type=int, default=None,
                        help="Trace a specific neuron index across layers")

    args = parser.parse_args()

    print("Model:", args.model)
    print("Mask Type:", args.mask_type)

    # Path setup (same as your version)
    MASK_DIR = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    # Other paths are unnecessary here (only mask is read)

    main()