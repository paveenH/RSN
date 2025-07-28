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


def get_samples(model, size, rsn_type, hs_dir, logits, base_dir):
    """
    Scan each task and load hidden states of samples where answers differ,
    stack them, and return pos_samples and neg_samples:
      pos_samples: np.ndarray, shape (N, L, D)
      neg_samples: np.ndarray, shape (N, L, D)
    """
    answer_file = f"answer_{rsn_type}_logits" if logits else f"answer_{rsn_type}"
    json_root = os.path.join(base_dir, answer_file, model)
    hs_root = os.path.join(hs_dir, model)
    all_pos = []
    all_neg = []

    for task in TASKS:
        # Construct file paths
        char_path = os.path.join(hs_root, f"{task}_{task}_{size}.npy")
        none_path = os.path.join(hs_root, f"{rsn_type}_{task}_{task}_{size}.npy")
        json_path = os.path.join(json_root, f"{task}_{size}_answers.json")

        # Skip if any file is missing
        if not (os.path.exists(char_path) and os.path.exists(none_path) and os.path.exists(json_path)):
            print("skip ", task)
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
            continue

        # Extract these samples
        sel_char = data_char[diff_indices, 0, ...]  # (k, L, D)
        sel_none = data_none[diff_indices, 0, ...]
        all_pos.append(sel_char)
        all_neg.append(sel_none)

    if not all_pos:
        raise RuntimeError("No inconsistent samples found!")

    pos = np.concatenate(all_pos, axis=0)  # (N, L, D)
    neg = np.concatenate(all_neg, axis=0)

    return pos, neg


def make_ttest_mask(pos, neg, percentage, localize_range, seed):
    """
    Perform t-test on pos/neg samples layer×unit,
    select top percentage% by absolute t-value,
    keep those positions in diff, set others to zero.
    """
    N, L, D = pos.shape
    total = L * D
    k = max(1, int((percentage / 100) * total))
    start, end = map(int, localize_range.split("-"))
    np.random.seed(seed)
    
    diff = np.mean(pos - neg, axis=0)  # (L, D)

    # 1) Calculate t-values of shape [L, D]
    t_vals = np.zeros((L, D), dtype=np.float32)
    for i in range(L):
        t_vals[i], _ = ttest_ind(np.abs(pos[:, i, :]), np.abs(neg[:, i, :]), axis=0, equal_var=False)
    # 2) Flatten + top-k
    flat = t_vals.flatten()
    idxs = np.argpartition(-np.abs(flat), k)[:k]
    mask_flat = np.zeros_like(flat, dtype=bool)
    mask_flat[idxs] = True
    mask = mask_flat.reshape((L, D))

    # 3) Construct mask_diff from diff using the mask
    mask_diff = np.zeros_like(diff, dtype=diff.dtype)  # diff shape = (L, D)
    mask_diff[mask] = diff[mask]

    # 4) Optionally remove embedding layer (layer 0)
    return mask_diff[1:, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diff-mask based on t-test")
    parser.add_argument("--model", required=True, help="Model name, e.g., llama3")
    parser.add_argument("--size", required=True, help="Model size, e.g., 8B")
    parser.add_argument("--type", default="non", help="non or none or non-")
    parser.add_argument("--logits", action="store_true", help="Use logits version of hidden states")
    parser.add_argument("--percentage", type=float, default=1.0, help="Retention ratio (%) e.g. 1.0 means top 1%")
    parser.add_argument("--localize_range", default="100-100", help="Layer range start-end (1-based)")
    parser.add_argument("--seed", type=int, default=42)
    
    
    args = parser.parse_args()
    
    base_dir = "/data2/paveen/RolePlaying/components"
    hidden_states_path = os.path.join(base_dir, f"hidden_states_{args.type}", args.model)
    
    # Select all inconsistent samples
    pos, neg = get_samples(
        model=args.model,
        size=args.size,
        rsn_type=args.type,
        hs_dir=hidden_states_path,
        logits=args.logits,
        base_dir=base_dir
    )
    
    
    print(f"Number of inconsistent samples: {pos.shape[0]}")
    print(f"Number of layers: {pos.shape[1]}, Hidden size: {pos.shape[2]}")



    #Generate t-test mask
    mask = make_ttest_mask(
        pos=pos, 
        neg=neg, 
        percentage=args.percentage, 
        localize_range=args.localize_range, 
        seed=args.seed)

    # 4) Save mask
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, mask)
    print("Mask saved to", args.output, "shape:", mask.shape)
