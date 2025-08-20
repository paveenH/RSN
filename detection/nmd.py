#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and save sparse masks (full-layer), using pre-computed mean hidden states.
This version saves a full (L-1, H) matrix (excluding embedding layer), without start/end cropping.
"""

import os
import numpy as np
import argparse


def get_nmd_mask(diff_char, diff_none, top_k):
    """
    Full-layer NMD-style mask.
    diff_char, diff_none: np.ndarray, shape (1, 1, L, H) including embedding at layer 0
    Returns: mask of shape (L-1, H), selecting top_k by |value| per layer (layers 1..L-1).
    """
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (L, H)
    Lm1, H = diff.shape
    mask = np.zeros_like(diff)
    k = max(1, top_k)
    for l in range(Lm1):
        vec = diff[l]
        # faster than full sort
        idxs = np.argpartition(np.abs(vec), -k)[-k:]
        mask[l, idxs] = vec[idxs]
    return mask


def get_random_mask(top_k, total_layers, hidden_dim, seed=42):
    """
    Full-layer RANDOM mask (positions ~ Uniform, values ∈ {-1, +1}).
    Returns: (L-1, H)
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((total_layers, hidden_dim))
    k = max(1, top_k)
    for l in range(total_layers):
        idxs = rng.choice(hidden_dim, k, replace=False)
        mask[l, idxs] = rng.choice([-1.0, 1.0], size=k)
    return mask


def get_diff_random_mask(diff_char, diff_none, top_k, seed=42):
    """
    Full-layer DIFF-RANDOM mask: random positions, but values taken from (diff_char - diff_none).
    Returns: (L-1, H)
    """
    rng = np.random.default_rng(seed)
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (L, H)
    Lm1, H = diff.shape

    mask = np.zeros_like(diff)
    k = max(1, top_k)
    for l in range(Lm1):
        idxs = rng.choice(H, k, replace=False)
        mask[l, idxs] = diff[l, idxs]
    return mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate FULL (L-1, H) sparse mask from mean hidden states")
    parser.add_argument("--model", type=str, default="stablelm", help="Model name")
    parser.add_argument("--size", type=str, default="12B", help="Model size label")
    parser.add_argument("--type", type=str, default="non", help="Type: 'non' or 'exp'")
    parser.add_argument("--hs", type=str, default="stablelm", help="Hidden state folder prefix")
    parser.add_argument("--percentage", type=float, default=0.01, help="Percentage of neurons to keep per layer")
    parser.add_argument("--logits", action="store_true", help="Use logits variant for HS_MEAN path")
    parser.add_argument("--mask_type", type=str, default="nmd",
                        choices=["nmd", "random", "diff_random"],
                        help="Which mask to save: nmd / random / diff_random")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for random/diff_random)")
    args = parser.parse_args()

    # Paths for HS means
    suffix = f"{args.hs}_{args.type}_logits" if args.logits else f"{args.hs}_{args.type}"
    HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{suffix}"
    diff_char = np.load(os.path.join(HS_MEAN, f"diff_mean_{args.size}.npy"))
    diff_none = np.load(os.path.join(HS_MEAN, f"none_diff_mean_{args.size}.npy"))

    # Shapes
    L, H = diff_char.squeeze(0).squeeze(0).shape
    Lm1 = L - 1  # exclude embedding layer 0
    # percentage is in %, e.g., 0.5 -> 0.5%
    top_k = max(1, int(H * args.percentage / 100))
    print(f"top_k per layer = {top_k}  (H={H}, percentage={args.percentage}%)")
    print(f"Total layers (incl. emb) L={L}; saved mask layers (excl. emb) L-1={Lm1}")

    # Build mask (full across all non-embedding layers)
    if args.mask_type == "nmd":
        mask = get_nmd_mask(diff_char, diff_none, top_k)
    elif args.mask_type == "random":
        mask = get_random_mask(top_k, Lm1, H, seed=args.seed)
    elif args.mask_type == "diff_random":
        mask = get_diff_random_mask(diff_char, diff_none, top_k, seed=args.seed)
    else:
        raise ValueError(f"Unknown mask_type: {args.mask_type}")

    print(f"{args.mask_type} Mask shape: {mask.shape}")  # (L-1, H)

    # Save (file name without start/end)
    mask_dir = f"/data2/paveen/RolePlaying/components/mask/{args.model}_{args.type}"
    if args.logits:
        mask_dir += "_logits"
    os.makedirs(mask_dir, exist_ok=True)

    mask_name = f"{args.mask_type}_{args.percentage}_{args.size}.npy"
    mask_path = os.path.join(mask_dir, mask_name)
    np.save(mask_path, mask)
    print(f"Saved FULL {args.mask_type} mask → {mask_path}")