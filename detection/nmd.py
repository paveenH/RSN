#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and save NMD sparse mask for editing, using pre-computed mean hidden states.
"""
import os
import numpy as np

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
    # ------------------ Configuration ------------------
    model = "stablelm"
    size = "12B"
    TYPE = "non"
    HS = "stablelm"  

    TOP_K = 17
    START_LAYER = 16
    END_LAYER = 22
    
    LOGITS = True

    # Choose mean path based on AnswerName/HS
    if LOGITS:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}_logits"
    else:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}"
    diff_char = np.load(f"{HS_MEAN}/diff_mean_{size}.npy")
    diff_none = np.load(f"{HS_MEAN}/none_diff_mean_{size}.npy")

    # Generate mask
    mask = get_nmd_mask(diff_char, diff_none, TOP_K, START_LAYER, END_LAYER)
    print(f"Mask shape: {mask.shape}")

    # Save
    MASK_SAVE_ROOT = f"/data2/paveen/RolePlaying/components/mask/{model}"
    os.makedirs(MASK_SAVE_ROOT, exist_ok=True)
    mask_fn = f"nmd_{TOP_K}_{START_LAYER}_{END_LAYER}_{size}.npy"
    mask_fp = os.path.join(MASK_SAVE_ROOT, mask_fn)
    np.save(mask_fp, mask)
    print(f"Saved NMD mask → {mask_fp}")