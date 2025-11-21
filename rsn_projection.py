#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute per-layer RSN projection scores for mediation analysis.

For each sample i and layer l:

    r_i^(l) = h_i^(l) · v_RSN^(l)

Output shape per task per role:
    (num_samples, num_layers)
"""

import os
import numpy as np
from tqdm import tqdm
from detection.task_list import TASKS

# ===================== CONFIG =====================
ROLES = ["non", "expert"]
MODEL = "llama3"
SIZE = "8B"

# dense mean diff (num_layers, hidden_dim)
HS_ROOT = "/data2/paveen/RolePlaying/components/hidden_states_non"
# DIFF_PATH = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{MODEL}_non_logits/diff_mean_{SIZE}.npy"
DIFF_PATH = "/data2/paveen/RolePlaying/components/mask/llama3_non_logits/nmd_0.5_1_33_8B.npy"
SAVE_DIR = f"/data2/paveen/RolePlaying/components/rsn_projection_layers/{MODEL}_{SIZE}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== Load dense RSN directions for all layers ======
diff = np.load(DIFF_PATH, allow_pickle=True)
diff = diff.squeeze()
num_layers, H = diff.shape
diff = diff[1:,:]
print(f"Final diff shape used = {diff.shape}")

# ====== detect tasks ======
for task_name in tqdm(TASKS, desc="Tasks"):
    print(f"\n==== Task: {task_name} ====")
    for role in ROLES:
        # choose filename based on role
        if role == "expert":
            filename = f"{task_name}_{task_name}_{SIZE}.npy"
        else:
            filename = f"non_{task_name}_{task_name}_{SIZE}.npy"
        hs_path = os.path.join(HS_ROOT, MODEL, filename)
        if not os.path.exists(hs_path):
            print(f"[WARN] missing hs for role={role}, task={task_name}")
            continue
        # load hidden states: shape (N, L, H)
        hs = np.load(hs_path)
        hs = hs.squeeze()
        N, L_hs, H_hs = hs.shape
        print(f"[{role}] hs shape = {hs.shape} (N={N}, L={L_hs}, H={H_hs})")

        if L_hs != num_layers:
            print(f"[ERROR] Hidden states L={L_hs} mismatch diff L={num_layers}")
        if H_hs != H:
            print(f"[ERROR] Hidden states H={H_hs} mismatch diff H={H}")
        layer_scores = np.sum(hs * diff[None, :, :], axis=-1)
        print(f"[{role}] → layer_scores shape = {layer_scores.shape}")
        # save
        out_path = os.path.join(SAVE_DIR, f"rsn_layers_{role}_{task_name}.npy")
        np.save(out_path, layer_scores)
        print(f"[{role}] saved → {out_path}")

print("\nSaved all per-layer RSN projection matrices to:", SAVE_DIR)