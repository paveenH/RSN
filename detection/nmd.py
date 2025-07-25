#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and save NMD sparse mask for editing, using pre-computed mean hidden states.
"""
import os
import numpy as np
import argparse


def get_nmd_mask(diff_char, diff_none, top_k, start, end):
    """
    diff_char, diff_none: np.ndarray, shape (1, 1, L, H) 
    Returns: mask of shape (L-1, H), masking out only top_k neurons per layer in [start, end).
    """
    diff = (diff_char - diff_none).squeeze(0).squeeze(0) 
    mask = np.zeros_like(diff)
    for l in range(diff.shape[0]):
        if start <= l < end:
            vec = diff[l]
            idxs = np.argsort(np.abs(vec))[-top_k:]
            mask[l, idxs] = vec[idxs]
    return mask[1:, :]  # remove embedding layer (layer 0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate NMD mask from mean hidden states")
    parser.add_argument("--model", type=str, default="stablelm", help="Model name")
    parser.add_argument("--size", type=str, default="12B", help="Model size label")
    parser.add_argument("--type", type=str, default="non", help="Type: 'non' or 'exp'")
    parser.add_argument("--hs", type=str, default="stablelm", help="Hidden state folder prefix")
    parser.add_argument("--top_k", type=int, default=17, help="Top-K neurons per layer to keep")
    parser.add_argument("--start_layer", type=int, default=16, help="Start layer index (inclusive)")
    parser.add_argument("--end_layer", type=int, default=22, help="End layer index (exclusive)")
    parser.add_argument("--logits", action="store_true", help="Use logits variant for HS_MEAN path")

    args = parser.parse_args()

    # Determine HS mean path
    suffix = f"{args.hs}_{args.type}_logits" if args.logits else f"{args.hs}_{args.type}"
    HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{suffix}"
    diff_char = np.load(os.path.join(HS_MEAN, f"diff_mean_{args.size}.npy"))
    diff_none = np.load(os.path.join(HS_MEAN, f"none_diff_mean_{args.size}.npy"))

    # Generate mask
    mask = get_nmd_mask(diff_char, diff_none, args.top_k, args.start_layer, args.end_layer)
    print(f"Mask shape: {mask.shape}")

    # Save
    mask_dir = f"/data2/paveen/RolePlaying/components/mask/{args.model}"
    os.makedirs(mask_dir, exist_ok=True)
    mask_name = f"nmd_{args.top_k}_{args.start_layer}_{args.end_layer}_{args.size}.npy"
    mask_path = os.path.join(mask_dir, mask_name)
    np.save(mask_path, mask)
    print(f"Saved NMD mask → {mask_path}")