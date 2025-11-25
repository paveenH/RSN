#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate NMD-style sparse masks over a layer range [start, end),
using pre-computed mean hidden states (diff_char, diff_none).

- Only layers in [start, end) are edited.
- Embedding layer (layer 0) is removed in the saved mask => shape (L-1, H).
- Supports three mask types: nmd / random / diff_random.
"""
import os
import numpy as np
import argparse


def get_nmd_mask(diff_char: np.ndarray, diff_none: np.ndarray, top_k: int, start: int, end: int) -> np.ndarray:
    """
    NMD: per-layer pick top_k by |diff| in [start, end).

    diff_char, diff_none: np.ndarray with shape (1, 1, L, H)
    Returns: mask with shape (L-1, H), removing layer 0 (embedding).
    """
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (L, H)
    L, H = diff.shape
    mask = np.zeros_like(diff)

    for l in range(L):
        if start <= l < end:
            vec = diff[l]
            idxs = np.argsort(np.abs(vec))[-top_k:]
            mask[l, idxs] = vec[idxs]
            
    return mask[1:, :]


def get_random_mask(top_k: int, start: int, end: int, total_layers: int, hidden_dim: int, seed: int = 42) -> np.ndarray:
    """
    RANDOM: random positions in [start, end), values in {-1, +1}.
    Returns: (L-1, H), removing embedding layer 0.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((total_layers, hidden_dim))
    k = max(1, int(top_k))

    for l in range(total_layers):
        if start <= l < end:
            idxs = rng.choice(hidden_dim, k, replace=False)
            mask[l, idxs] = rng.choice([-1.0, 1.0], size=k)

    return mask[1:, :]


def get_diff_random_mask(
    diff_char: np.ndarray, diff_none: np.ndarray, top_k: int, start: int, end: int, seed: int = 42
) -> np.ndarray:
    """
    DIFF-RANDOM: random positions in [start, end), but values taken
    from the true diff (diff_char - diff_none) at those positions.

    Returns: (L-1, H), removing embedding layer 0.
    """
    rng = np.random.default_rng(seed)
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (L, H)
    L, H = diff.shape
    mask = np.zeros_like(diff)

    k = max(1, int(top_k))
    for l in range(L):
        if start <= l < end:
            idxs = rng.choice(H, k, replace=False)
            mask[l, idxs] = diff[l, idxs]

    return mask[1:, :]


def get_sparse_fv_mask(diff_char: np.ndarray, diff_none: np.ndarray,
                       top_k: int, start: int, end: int) -> np.ndarray:
    """
    Sparse FV: pick top_k neurons ACROSS ALL LAYERS in [start, end),
    instead of per-layer selection.

    This treats the diff[start:end, :] region as one big vector
    with shape ((end-start) * H), selects top_k by |diff| globally,
    and returns a mask of shape (L-1, H), removing embedding layer.

    diff_char, diff_none: (1, 1, L, H)
    """
    # (L, H)
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)
    L, H = diff.shape

    # extract block to search over: (num_layers, H)
    block = diff[start:end]                               # (end-start, H)
    flat = np.abs(block).reshape(-1)                      # flatten

    k = max(1, int(top_k))                                # safety
    # top_k indices (unsorted)
    idxs = np.argpartition(-flat, k-1)[:k]                # (k,)

    # convert flat indices → 2D indices (layer offset, neuron id)
    layer_offsets = idxs // H
    neuron_ids = idxs % H

    # build full mask
    mask = np.zeros_like(diff)
    for lo, nid in zip(layer_offsets, neuron_ids):
        real_layer = start + lo
        mask[real_layer, nid] = diff[real_layer, nid]

    # remove embedding layer (0)
    return mask[1:, :]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate NMD/diff_random/random masks from mean hidden states")
    parser.add_argument("--model", type=str, default="stablelm", help="Model name")
    parser.add_argument("--size", type=str, default="12B", help="Model size label")
    parser.add_argument("--type", type=str, default="non", help="Type: 'non' or 'exp'")
    parser.add_argument("--hs", type=str, default="stablelm", help="Hidden state folder prefix")
    parser.add_argument("--percentage", type=float, default=0.01, help="Percentage (in %) of neurons to keep per layer, e.g. 0.5 for 0.5%")
    parser.add_argument("--start_layer", type=int, default=16, help="Start layer index (inclusive)")
    parser.add_argument("--end_layer", type=int, default=22, help="End layer index (exclusive)")
    parser.add_argument("--logits", action="store_true", help="Use logits variant for HS_MEAN path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for random/diff_random)")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="nmd",
        choices=["nmd", "random", "diff_random", "sparse_fv"],
        help="Which mask to save: nmd / random / diff_random",
    )

    args = parser.parse_args()

    # Determine HS mean path
    suffix = f"{args.hs}_{args.type}_logits" if args.logits else f"{args.hs}_{args.type}"
    HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{suffix}"
    diff_char = np.load(os.path.join(HS_MEAN, f"diff_mean_{args.size}.npy"))
    diff_none = np.load(os.path.join(HS_MEAN, f"none_diff_mean_{args.size}.npy"))

    # Shapes & top-k
    total_layers, hidden_dim = diff_char.squeeze(0).squeeze(0).shape
    top_k = max(1, int(hidden_dim * args.percentage / 100.0))
    print(f"Hidden size H={hidden_dim}, percentage={args.percentage}% -> top_k per layer = {top_k}")
    print(f"Total layers (INCLUDING embedding layer 0) L={total_layers}")
    print(f"Edit range: [{args.start_layer}, {args.end_layer})")

    # Build mask (returns shape (L-1, H), embedding removed)
    if args.mask_type == "nmd":
        mask = get_nmd_mask(diff_char, diff_none, top_k, args.start_layer, args.end_layer)
    elif args.mask_type == "random":
        mask = get_random_mask(top_k, args.start_layer, args.end_layer, total_layers, hidden_dim, seed=args.seed)
    elif args.mask_type == "diff_random":
        mask = get_diff_random_mask(diff_char, diff_none, top_k, args.start_layer, args.end_layer, seed=args.seed)
    elif args.mask_type == "sparse_fv":
        mask = get_sparse_fv_mask(diff_char, diff_none, top_k, args.start_layer, args.end_layer)
    else:
        raise ValueError(f"Unknown mask_type: {args.mask_type}")

    print(f"{args.mask_type} Mask shape: {mask.shape}")  # (L-1, H)

    # Save
    mask_dir = f"/data2/paveen/RolePlaying/components/mask/{args.model}_{args.type}"
    if args.logits:
        mask_dir += "_logits"
    os.makedirs(mask_dir, exist_ok=True)

    mask_name = f"{args.mask_type}_{args.percentage}_{args.start_layer}_{args.end_layer}_{args.size}.npy"
    mask_path = os.path.join(mask_dir, mask_name)
    np.save(mask_path, mask)
    print(f"Saved {args.mask_type} mask → {mask_path}")
