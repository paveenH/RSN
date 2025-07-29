#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:16:24 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from task_list import TASKS
from scipy.stats import ttest_ind


def get_samples(model, size, rsn_type, hs_root, json_root):
    """
    Scan each task and load hidden states of samples where answers differ,
    stack them, and return pos_samples and neg_samples:
      pos_samples: np.ndarray, shape (N, L, D)
      neg_samples: np.ndarray, shape (N, L, D)
    """

    all_pos = []
    all_neg = []
    total_count = 0

    for task in TASKS:
        # Construct file paths
        char_path = os.path.join(hs_root, f"{task}_{task}_{size}.npy")
        none_path = os.path.join(hs_root, f"{rsn_type}_{task}_{task}_{size}.npy")
        json_path = os.path.join(json_root, f"{task}_{size}_answers.json")

        # Skip if any file is missing
        if not (os.path.exists(char_path) and os.path.exists(none_path) and os.path.exists(json_path)):
            print(f"[Skip] Missing file for task: {task}") 
            continue

        data_char = np.load(char_path)
        data_none = np.load(none_path)
        # May be (N,33,D) or (N,1,33,D)
        if data_char.ndim == 3:
            data_char = data_char[:, None, ...]
        if data_none.ndim == 3:
            data_none = data_none[:, None, ...]

        # Load JSON, select sample indices where answers differ
        with open(json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        diff_indices = []
        for idx, entry in enumerate(j.get("data", [])):
            a1 = entry.get(f"answer_non_{task}")
            a2 = entry.get(f"answer_{task}")
            if a1 != a2:
                diff_indices.append(idx)
        if not diff_indices:
            print(f"[Info] No differing samples in task: {task}")
            continue

        # Extract these samples
        sel_char = data_char[diff_indices, 0, ...]  # (k, L, D)
        sel_none = data_none[diff_indices, 0, ...]
        all_pos.append(sel_char)
        all_neg.append(sel_none)
        
        print(f"Task: {task}] Inconsistent samples found: {len(diff_indices)}") 
        total_count += len(diff_indices)

    if not all_pos:
        raise print("No inconsistent samples found!")

    pos = np.concatenate(all_pos, axis=0)  # (N, L, D)
    neg = np.concatenate(all_neg, axis=0)
    
    print(f"[Done] Total inconsistent samples: {total_count}")  
    print(f"[Shape] pos: {pos.shape}, neg: {neg.shape}")  

    return pos, neg


def is_topk(a, k=1):
    """
    Top-k selection, considering sign.
    """
    _, rix = np.unique(-a, return_inverse=True)
    return (rix < k).astype(int).reshape(a.shape)


def is_topk_abs(a, k=1):
    """
    Top-k selection by absolute value.
    """
    flat = np.abs(a).flatten()
    idxs = np.argpartition(-flat, k)[:k]
    mask = np.zeros_like(flat, dtype=int)
    mask[idxs] = 1
    return mask.reshape(a.shape)


def make_ttest_mask(pos, neg, percentage, start, end, use_abs=False):
    """
    Perform t-test on pos/neg samples layer×unit,
    select top percentage% by absolute t-value,
    keep those positions in diff, set others to zero.
    """
    N, L, D = pos.shape

    num_sel_layers = end - start    # e.g. 1-33 → 32
    total = num_sel_layers * D
    k = max(1, int((percentage / 100) * total))
    print ("[INFO] total selected neurons: ", k)

    diff = np.mean(pos - neg, axis=0)  # (L, D)
    t_vals = np.zeros((L, D), dtype=np.float32)

    for i in range(start, end):
        pos_i = pos[:, i, :]
        neg_i = neg[:, i, :]
        if use_abs:
            t_vals[i], _ = ttest_ind(np.abs(pos_i), np.abs(neg_i), axis=0, equal_var=False)
        else:
            t_vals[i], _ = ttest_ind(pos_i, neg_i, axis=0, equal_var=False)

    t_block = t_vals[start:end].reshape(-1)  # (num_sel_layers * D,)
    if use_abs:
        mask_block = is_topk(t_block, k)
    else:
        mask_block = is_topk_abs(t_block, k)

    mask_block = mask_block.reshape((num_sel_layers, D))  # (end-start, D)
    print(np.sum(mask_block))
    mask = np.zeros((L, D), dtype=int)
    print(np.sum(mask))
    mask[start:end] = mask_block

    mask_diff = np.zeros_like(diff, dtype=diff.dtype)
    mask_diff[mask.astype(bool)] = diff[mask.astype(bool)]
    print(np.sum(mask_diff != 0))

    return mask_diff[1:, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diff-mask based on t-test")
    parser.add_argument("--model", required=True, help="Model name, e.g., llama3")
    parser.add_argument("--size", required=True, help="Model size, e.g., 8B")
    parser.add_argument("--type", default="non", help="non or none or non-")
    parser.add_argument("--logits", action="store_true", help="Use logits version of hidden states")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--percentage", type=float, default=1.0, help="Retention ratio (%) e.g. 1.0 means top 1%")
    parser.add_argument("--layer", default="1-33", help="Layer range start-end (1-based)")

    args = parser.parse_args()

    # Path
    base_dir = "/data2/paveen/RolePlaying/components"
    answer_file = f"answer_{args.type}_logits" if args.logits else f"answer_{args.type}"
    json_root = os.path.join(base_dir, answer_file, args.model)
    hs_root = os.path.join(base_dir, f"hidden_states_{args.type}", args.model)

    # Select all inconsistent samples
    pos, neg = get_samples(args.model, args.size, args.type, hs_root, json_root)

    print(f"Number of inconsistent samples: {pos.shape[0]}")
    print(f"Number of layers: {pos.shape[1]}, Hidden size: {pos.shape[2]}")

    # Generate t-test mask
    start, end = map(int, args.layer.split("-"))
    mask = make_ttest_mask(pos, neg, args.percentage, start, end, args.abs)

    # Save mask
    mask_dir = f"/data2/paveen/RolePlaying/components/mask/{args.model}"
    os.makedirs(mask_dir, exist_ok=True)
    
    if args.abs:
        mask_name = f"ttest_{args.percentage}_{start}_{end}_{args.size}_abs.npy"
    else:
        mask_name = f"ttest_{args.percentage}_{start}_{end}_{args.size}.npy"
    mask_path = os.path.join(mask_dir, mask_name)
    
    np.save(mask_path, mask)
    print("Mask saved to", mask_path, "shape:", mask.shape)
