#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:18:42 2025

@author: paveenhuang

Adding row-based baseline correction for L2_diff_norm and cos_change.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

TASKS = [
    "abstract_algebra",
    # "anatomy",
    # "econometrics",
    # "global_facts",
    # "jurisprudence"
]

model = "llama3"
size = "8B"
data_org_dir = os.path.join(os.getcwd(), "llama3_v3_mdf")
data_rpl_dir = os.path.join(os.getcwd(), "llama3_v3_rpl2")
plot_dir = os.path.join(os.getcwd(), "plot", "llama3_v3_rpl2")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for task in TASKS:
    none_org_path = os.path.join(data_org_dir, f"none_{task}_{model}_{size}_org.npy")
    mdf_path = os.path.join(data_rpl_dir, f"none_{task}_{model}_{size}_rpl.npy")
    
    if not os.path.exists(none_org_path):
        print(f"[Error] Original file for {task} not found: {none_org_path}")
        continue
    if not os.path.exists(mdf_path):
        print(f"[Error] Modified file for {task} not found: {mdf_path}")
        continue

    none_org = np.load(none_org_path)      # shape: (33, 4096)
    none_org = none_org[1:, ]            # remove embedding layer -> (32, hidden_size)
    mdf_mean = np.load(mdf_path)         # shape: (31, 33, 4096)
    mdf_mean = mdf_mean[:, 1:, :]        # -> (31, 32, hidden_size)

    print(f"\nTask: {task}")
    print("none_org shape:", none_org.shape)   # (32, hidden_size)
    print("mdf_mean shape:", mdf_mean.shape)   # (31, 32, hidden_size)

    # For broadcasting
    none_org_expanded = none_org[np.newaxis, :, :]  # shape (1, 32, hidden_size) => (31, 32, hidden_size) after broadcast

    # ============== 1) Normalized L2 Difference ==============
    diff = mdf_mean - none_org_expanded  # (31, 32, hidden_size)
    l2_diff = np.linalg.norm(diff, axis=-1)  # (31, 32)

    layer_norms = np.linalg.norm(none_org, axis=-1)  # shape (32,)
    l2_diff_norm = l2_diff / layer_norms[np.newaxis, :]

    # for i in range(l2_diff_norm.shape[0]):
    #     if i < 30:
    #         baseline_value = l2_diff_norm[i+1, i] 
    #     else:
    #         baseline_value = 0.0
    #     l2_diff_norm[:, i] -= baseline_value
    # l2_diff_norm = np.maximum(l2_diff_norm, 0)

    # ============== 2) Cosine Similarity Change ==============
    dot_product = np.sum(mdf_mean * none_org_expanded, axis=-1)  # (31, 32)
    norm_mdf = np.linalg.norm(mdf_mean, axis=-1)  # (31, 32)
    norm_none = np.linalg.norm(none_org, axis=-1) # (32,)
    cos_sim = dot_product / (norm_mdf * norm_none[np.newaxis, :])  # (31, 32)
    cos_change = 1 - cos_sim  # (31, 32)

    # for i in range(cos_change.shape[0]):
    #     if i < 30:
    #         baseline_value = cos_change[i+1, i]
    #     else:
    #         baseline_value = 0.0
    #     cos_change[:, i] -= baseline_value
    # cos_change = np.maximum(cos_change, 0)

    # ============== 3) Visualize ==============
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.edgecolor'] = '#333333'
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10

    # For final layer's row: shape => (31,)
    final_layer_impact_l2 = l2_diff_norm[:, -1]  # row dimension=31, col=-1 => last layer
    final_layer_cos_change = cos_change[:, -1]

    # ------ Plot L2 Heatmap ------
    plt.figure(figsize=(12, 4))
    plt.imshow(l2_diff_norm.T, aspect='auto', cmap='viridis',
               interpolation='nearest', origin='upper')
    plt.xticks(np.arange(0, 31, 5), [str(x+1) for x in np.arange(0, 31, 5)])
    plt.yticks(np.arange(0, 32, 4), [str(y+1) for y in np.arange(0, 32, 4)])
    plt.xlabel("Replacement Experiment Index", fontsize=11, fontweight='medium')
    plt.ylabel("Modified Layer Index", fontsize=11, fontweight='medium')
    task_name = task.replace("_", " ")
    plt.title(f"Layer-wise Impact (Normalized L2) for {task_name}", fontsize=12, fontweight='bold', pad=10)
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Normalized L2 Impact (Corrected)', size=10)
    plt.xlim(-0.5, 30.5)
    plt.ylim(-0.5, 31.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{task}_layer_impact_l2_corrected.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ------ Plot Cosine Change Heatmap ------
    plt.figure(figsize=(12, 4))
    plt.imshow(cos_change.T, aspect='auto', cmap='plasma',
               interpolation='nearest', origin='upper')
    plt.xticks(np.arange(0, 31, 5), [str(x+1) for x in np.arange(0, 31, 5)])
    plt.yticks(np.arange(0, 32, 5), [str(y+1) for y in np.arange(0, 32, 5)])
    plt.xlabel("Replacement Experiment Index", fontsize=11, fontweight='medium')
    plt.ylabel("Modified Layer Index", fontsize=11, fontweight='medium')
    plt.title(f"Layer-wise Impact (Cosine Change) for {task_name}", fontsize=12, fontweight='bold', pad=10)
    cbar2 = plt.colorbar(pad=0.01)
    cbar2.ax.tick_params(labelsize=9)
    cbar2.set_label('Cosine Change (Corrected)', size=10)
    plt.xlim(-0.5, 30.5)
    plt.ylim(-0.5, 31.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{task}_layer_impact_cos_corrected.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ------ Combined Impact Plot ------
    x_coords = np.arange(1, 32)
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, final_layer_impact_l2, 'o-', color='#1f77b4', label="L2 Impact (Corrected)", linewidth=1.5, markersize=5)
    plt.plot(x_coords, final_layer_cos_change, 's-', color='#ff7f0e', label="Cosine Change (Corrected)", linewidth=1.5, markersize=5)
    plt.xlabel("Modified Layer Index", fontsize=11, fontweight='medium')
    plt.ylabel("Final Layer Impact", fontsize=11, fontweight='medium')
    plt.title(f"Impact on Final Layer (Corrected) for {task_name}", fontsize=12, fontweight='bold', pad=10)
    plt.xticks(np.arange(1, 33, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{task}_combined_impact_corrected.png"), dpi=300, bbox_inches='tight')
    plt.show()