#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified mask generator:
- ttest / ttest_abs
- dense_pca
- sparse_pca
- global_sparse_pca
- selection_pca (Hybrid: PCA Select + Diff Value)
"""

import os
import json
import argparse
import numpy as np
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from task_list import TASKS

# ─────────────────────────────
# Load samples
# ─────────────────────────────
def get_samples(model, size, rsn_type, hs_root, json_root):
    all_pos, all_neg = [], []
    total = 0

    for task in TASKS:
        char_path = os.path.join(hs_root, f"{task}_{task}_{size}.npy")
        # Ensure rsn_type matches your file naming convention (e.g., 'non')
        none_path = os.path.join(hs_root, f"{rsn_type}_{task}_{task}_{size}.npy")
        json_path = os.path.join(json_root, f"{task}_{size}_answers.json")

        if not (os.path.exists(char_path) and os.path.exists(none_path) and os.path.exists(json_path)):
            print(f"[Skip] Missing files for task {task}")
            continue

        try:
            # Using mmap_mode='r' can save memory if files are huge, 
            # but standard load is faster if memory permits.
            data_char = np.load(char_path)
            data_none = np.load(none_path)
        except Exception as e:
            print(f"[Error] Failed loading {task}: {e}")
            continue

        if data_char.ndim == 3:
            data_char = data_char[:, None, ...]
        if data_none.ndim == 3:
            data_none = data_none[:, None, ...]

        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)

        diff_idx = [
            i for i, entry in enumerate(j["data"])
            if entry.get(f"answer_non_{task}") != entry.get(f"answer_{task}")
        ]

        if not diff_idx:
            continue

        all_pos.append(data_char[diff_idx, 0])
        all_neg.append(data_none[diff_idx, 0])
        total += len(diff_idx)
        print(f"[Task {task}] Inconsistent: {len(diff_idx)}")

    if not all_pos:
        raise ValueError("No inconsistent samples found!")

    pos = np.concatenate(all_pos, axis=0)
    neg = np.concatenate(all_neg, axis=0)

    print(f"[Done] Total inconsistent = {total}")
    print(f"[Shapes] pos={pos.shape}, neg={neg.shape}")
    return pos, neg

def save_mask(mask, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, name)
    np.save(out_path, mask)
    print(f"[SAVE] Mask saved to {out_path}, shape={mask.shape}")

# ─────────────────────────────
# T-test mask
# ─────────────────────────────
def make_ttest_mask(pos, neg, percentage, start, end, abs_mode=False):
    pos = pos.astype(np.float64)
    neg = neg.astype(np.float64)

    N, L, D = pos.shape
    Lr = end - start
    total_units = Lr * D
    top_k = max(1, int(total_units * percentage / 100))
    print(f"[TTEST] Select top-{top_k} / {total_units} units ({percentage}%) abs_mode={abs_mode}")

    diff = np.mean(pos - neg, axis=0)

    t_vals = np.zeros((Lr, D), dtype=np.float64)

    for li, layer in enumerate(range(start, end)):
        p = pos[:, layer, :]
        n = neg[:, layer, :]

        if abs_mode:
            p = np.abs(p)
            n = np.abs(n)

        # equal_var=False performs Welch's t-test
        t_layer, _ = ttest_ind(p, n, axis=0, equal_var=False)
        
        # [Safety] Replace NaNs with 0 to prevent sorting crashes
        t_layer = np.nan_to_num(t_layer, nan=0.0)
        
        t_vals[li] = t_layer

    flat = np.abs(t_vals).reshape(-1)
    idxs = np.argpartition(-flat, top_k - 1)[:top_k]

    sparse = np.zeros_like(t_vals)
    sparse_flat = sparse.reshape(-1)
    sparse_flat[idxs] = 1
    sparse = sparse_flat.reshape(Lr, D).astype(bool)

    full = np.zeros((L, D))
    full[start:end][sparse] = diff[start:end][sparse]

    return full[1:, :]


# ─────────────────────────────
# T-test Layer-wise mask (New)
# ─────────────────────────────
def make_ttest_layerwise_mask(pos, neg, percentage, start, end, abs_mode=False):
    """
    Select top-k neurons PER LAYER based on t-statistics.
    Value injected is still the mean difference (Delta mu).
    """
    # Cast for safety
    pos = pos.astype(np.float64)
    neg = neg.astype(np.float64)

    N, L, D = pos.shape
    Lr = end - start

    # Calculate Top-k per layer (relative to D, not Total Units)
    top_k = max(1, int(D * percentage / 100))
    print(f"[TTEST-LAYER] Select per-layer top-{top_k} / {D} units ({percentage}%) abs_mode={abs_mode}")

    # 1. Calculate values to inject (Mean Shift)
    diff = np.mean(pos - neg, axis=0)  # (L, D)

    # 2. Calculate selection criteria (T-values)
    sparse = np.zeros((Lr, D), dtype=np.float64)
    
    for li, layer in enumerate(range(start, end)):
        p = pos[:, layer, :]
        n = neg[:, layer, :]

        if abs_mode:
            p = np.abs(p)
            n = np.abs(n)

        # Welch's t-test
        t_vals, _ = ttest_ind(p, n, axis=0, equal_var=False)
        t_vals = np.nan_to_num(t_vals, nan=0.0) # Safety

        # 3. Per-layer Selection
        # Sort by absolute t-value magnitude
        idx = np.argsort(-np.abs(t_vals))[:top_k]
        
        # 4. Inject Mean Difference values
        sparse[li, idx] = diff[layer, idx]

    # Build full mask
    full = np.zeros((L, D))
    full[start:end] = sparse

    # Remove embedding layer
    return full[1:, :]

# ─────────────────────────────
# Shared PCA computation
# ─────────────────────────────
def compute_pca_block(pos, neg, start, end):
    diff = pos - neg
    block = diff[:, start:end, :]
    N, Lr, D = block.shape
    print(f"[PCA] PCA on shape N={N}, Lr={Lr}, D={D}")

    X = block.reshape(N, Lr * D)
    
    # [Safety] Fixed random_state for reproducibility
    pca = PCA(n_components=1, random_state=42)
    v = pca.fit(X).components_[0]
    vec_block = v.reshape(Lr, D)
    return vec_block, Lr, D

# ─────────────────────────────
# PCA variants
# ─────────────────────────────
def dense_pca_mask(pos, neg, start, end, **kwargs):
    vec_block, Lr, D = compute_pca_block(pos, neg, start, end)
    L = pos.shape[1]
    full = np.zeros((L, D))
    full[start:end] = vec_block
    return full[1:, :]

def global_sparse_pca_mask(pos, neg, start, end, percentage, **kwargs):
    vec_block, Lr, D = compute_pca_block(pos, neg, start, end)
    L = pos.shape[1]
    total_units = Lr * D
    k_global = max(1, int(total_units * percentage / 100))

    flat = np.abs(vec_block).reshape(-1)
    idxs = np.argpartition(-flat, k_global - 1)[:k_global]

    sparse = np.zeros_like(vec_block)
    layer_idx = idxs // D
    neuron_idx = idxs % D

    for l, n in zip(layer_idx, neuron_idx):
        sparse[l, n] = vec_block[l, n]

    full = np.zeros((L, D))
    full[start:end] = sparse
    return full[1:, :]

def sparse_pca_mask(pos, neg, start, end, percentage, **kwargs):
    vec_block, Lr, D = compute_pca_block(pos, neg, start, end)
    top_k = max(1, int(percentage / 100 * D))
    print(f"[Sparse PCA] Per-layer top-{top_k} (percentage={percentage}%)")

    L = pos.shape[1]
    sparse = np.zeros_like(vec_block)

    for l in range(Lr):
        row = vec_block[l]
        idx = np.argsort(-np.abs(row))[:top_k]
        sparse[l, idx] = row[idx]

    full = np.zeros((L, D))
    full[start:end] = sparse
    return full[1:, :]

def pca_selection_mask(pos, neg, start, end, percentage, **kwargs):
    """
    PCA-based neuron selection (per-layer top-k), but apply Δh values.
    """
    # Clip to prevent overflow if values are huge (optional safety)
    pos = np.clip(pos, -1e6, 1e6).astype(np.float64)
    neg = np.clip(neg, -1e6, 1e6).astype(np.float64)
    
    # 1. Calculate values to inject (Mean Shift)
    diff = np.mean(pos - neg, axis=0)

    # 2. Calculate selection criteria (PCA Variance)
    vec_block, Lr, D = compute_pca_block(pos, neg, start, end)
    L = pos.shape[1]

    top_k = max(1, int(percentage / 100 * D))
    print(f"[Selection PCA] Per-layer top-{top_k} (percentage={percentage}%)")

    sparse = np.zeros_like(vec_block)

    for l in range(Lr):
        row = vec_block[l]
        # Select indices based on PCA magnitude
        idx = np.argsort(-np.abs(row))[:top_k]
        # Inject values from Difference Matrix
        sparse[l, idx] = diff[start + l, idx]

    full = np.zeros((L, D))
    full[start:end] = sparse
    return full[1:, :]

# Mapping
MASK_FUNCS = {
    "ttest": lambda pos, neg, start, end, percentage: make_ttest_mask(pos, neg, percentage, start, end, abs_mode=False),
    "ttest_abs": lambda pos, neg, start, end, percentage: make_ttest_mask(pos, neg, percentage, start, end, abs_mode=True),
    "ttest_layer": lambda pos, neg, start, end, percentage: make_ttest_layerwise_mask(pos, neg, percentage, start, end, abs_mode=False),
    "ttest_layer_abs": lambda pos, neg, start, end, percentage: make_ttest_layerwise_mask(pos, neg, percentage, start, end, abs_mode=True),
    "dense_pca": lambda pos, neg, start, end, percentage: dense_pca_mask(pos, neg, start, end),
    "sparse_pca": lambda pos, neg, start, end, percentage: sparse_pca_mask(pos, neg, start, end, percentage),
    "global_sparse_pca": lambda pos, neg, start, end, percentage: global_sparse_pca_mask(pos, neg, start, end, percentage),
    "selection_pca": lambda pos, neg, start, end, percentage: pca_selection_mask(pos, neg, start, end, percentage),
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--type", default="non")
    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--mask_type", required=True,
        choices=["ttest", "ttest_abs", "ttest_layer", "ttest_layer_abs",
                 "dense_pca", "sparse_pca", "global_sparse_pca", "selection_pca"])
    parser.add_argument("--percentage", type=float, default=1.0)
    parser.add_argument("--layer", default="1-33")

    args = parser.parse_args()

    base = "/data2/paveen/RolePlaying/components"
    ans_dir = f"answer_{args.type}_logits" if args.logits else f"answer_{args.type}"
    json_root = os.path.join(base, ans_dir, args.model)
    hs_root = os.path.join(base, f"hidden_states_{args.type}", args.model)

    save_dir = f"/data2/paveen/RolePlaying/components/mask/{args.model}_{args.type}"
    if args.logits:
        save_dir += "_logits"
        
    print(f"\n================ PATH CHECK ================\n")
    print(f"MODEL           : {args.model}")
    print(f"MASK TYPE       : {args.mask_type}")
    print(f"PERCENTAGE      : {args.percentage}%")
    print(f"LAYER RANGE     : {args.layer}")
    print(f"SAVE DIR        : {save_dir}")
    print("============================================\n")

    pos, neg = get_samples(args.model, args.size, args.type, hs_root, json_root)
    start, end = map(int, args.layer.split("-"))

    print(f"\n[MASK TYPE] {args.mask_type}")
    
    mask = MASK_FUNCS[args.mask_type](pos, neg, start, end, args.percentage)
    name = f"{args.mask_type}_{args.percentage}_{start}_{end}_{args.size}.npy"
    mask_path = os.path.join(save_dir, name)

    print(f"[SAVE PATH] → {mask_path}\n")
    save_mask(mask, save_dir, name)