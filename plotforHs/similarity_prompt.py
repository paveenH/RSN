#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated script to:
1) Compute difference-based metrics (Dice, Jaccard, Pearson, etc.)
2) Plot the representational overlap (Dice vs. top N)
3) Compare similarity metrics (Pearson, Spearman, Cosine, Wasserstein, Euclidean)
   across layers for llama3_v3 vs. llama3_v5 with a common y-scale.
Created on Thu Mar  6 17:58:32 2025

@author: paveen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 1) Set publication style and define parameters
# -------------------------------------------------------------------
sns.set_theme(style="white", context="paper", font_scale=1.2)

path = os.getcwd()
model_v3 = "llama3_v3"
model_v5 = "llama3_v5"
size = "8B"

data_path_v3 = os.path.join(path, model_v3)
data_path_v5 = os.path.join(path, model_v5)

file_char_v3 = os.path.join(data_path_v3, f"all_mean_{size}.npy")
file_none_char_v3 = os.path.join(data_path_v3, f"none_all_mean_{size}.npy")

file_char_v5 = os.path.join(data_path_v5, f"all_mean_{size}.npy")
file_none_char_v5 = os.path.join(data_path_v5, f"none_all_mean_{size}.npy")

save_dir = os.path.join(path, "plot", "comparison_v3_vs_v5")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# -------------------------------------------------------------------
# 2) Load data for v3 and v5
# -------------------------------------------------------------------
data_char_v3 = np.load(file_char_v3)
data_none_char_v3 = np.load(file_none_char_v3)
data_char_v5 = np.load(file_char_v5)
data_none_char_v5 = np.load(file_none_char_v5)

print("Loaded shapes:")
print("v3 char:", data_char_v3.shape, " v3 none_char:", data_none_char_v3.shape)
print("v5 char:", data_char_v5.shape, " v5 none_char:", data_none_char_v5.shape)

# -------------------------------------------------------------------
# 3) Compute difference (v3: char - none_char, v5: char - none_char)
#    Exclude layer 0 (embedding layer) for difference-based analysis
# -------------------------------------------------------------------
data_char_diff_v3 = data_char_v3 - data_none_char_v3  # shape: (samples, 1, layers, hidden_size)
data_char_diff_v5 = data_char_v5 - data_none_char_v5
char_diff_v3 = data_char_diff_v3[:, :, 1:, :]  # exclude layer 0
char_diff_v5 = data_char_diff_v5[:, :, 1:, :]
# char_diff_v3 = data_char_diff_v3
# char_diff_v5 = data_char_diff_v5
# char_diff_v3 = data_char_v3
# char_diff_v5 = data_char_v5
# Compute mean difference per neuron for each layer
mean_diff_v3 = np.mean(char_diff_v3, axis=(0, 1))  
mean_diff_v5 = np.mean(char_diff_v5, axis=(0, 1))
num_layers_diff, hidden_size_diff = mean_diff_v3.shape

# # -------------------------------------------------------------------
# 4) Compute difference-based metrics (Dice, Jaccard, Pearson, etc.)
# -------------------------------------------------------------------
top_neurons = 20
dice_list, jaccard_list = [], []
pearson_list, spearman_list = [], []
cosine_list = []
kl_list, wasserstein_list, wasserstein_norm_list, euclidean_list = [], [], [], []

def dice_coefficient(set1, set2):
    intersection = len(set1.intersection(set2))
    denom = (len(set1) + len(set2))
    return 2 * intersection / denom if denom > 0 else np.nan

for layer_idx in range(num_layers_diff):
    v3_layer = mean_diff_v3[layer_idx]
    v5_layer = mean_diff_v5[layer_idx]

    # Top-N indices (by absolute value)
    top_v3 = set(np.argsort(np.abs(v3_layer))[-top_neurons:])
    top_v5 = set(np.argsort(np.abs(v5_layer))[-top_neurons:])

    # Dice & Jaccard
    dice_list.append(dice_coefficient(top_v3, top_v5))
    union_set = top_v3.union(top_v5)
    intersection_set = top_v3.intersection(top_v5)
    jacc = len(intersection_set) / len(union_set) if len(union_set) > 0 else np.nan
    jaccard_list.append(jacc)

    # Pearson & Spearman
    pearson_val = np.corrcoef(v3_layer, v5_layer)[0, 1]
    spearman_val = stats.spearmanr(v3_layer, v5_layer)[0]
    pearson_list.append(pearson_val)
    spearman_list.append(spearman_val)

    # Cosine
    cos_val = cosine_similarity(v3_layer.reshape(1, -1), v5_layer.reshape(1, -1))[0, 0]
    cosine_list.append(cos_val)

    # KL Divergence (Softmax normalization on abs values)
    abs_v3 = np.exp(np.abs(v3_layer)); abs_v3 /= abs_v3.sum()
    abs_v5 = np.exp(np.abs(v5_layer)); abs_v5 /= abs_v5.sum()
    kl_sym = (stats.entropy(abs_v3, abs_v5) + stats.entropy(abs_v5, abs_v3)) / 2
    kl_list.append(kl_sym)

    # Wasserstein (unnormalized)
    w_dist = stats.wasserstein_distance(v3_layer, v5_layer)
    wasserstein_list.append(w_dist)

    # Wasserstein (Z-score)
    v3_mean, v3_std = v3_layer.mean(), v3_layer.std()
    v5_mean, v5_std = v5_layer.mean(), v5_layer.std()
    v3_norm = (v3_layer - v3_mean) / v3_std if v3_std > 0 else (v3_layer - v3_mean)
    v5_norm = (v5_layer - v5_mean) / v5_std if v5_std > 0 else (v5_layer - v5_mean)
    w_norm = stats.wasserstein_distance(v3_norm, v5_norm)
    wasserstein_norm_list.append(w_norm)

    # Euclidean Distance (Unit Vector Normalized)
    v3_unit = v3_layer / np.linalg.norm(v3_layer)
    v5_unit = v5_layer / np.linalg.norm(v5_layer)
    euc = np.linalg.norm(v3_unit - v5_unit)
    euclidean_list.append(euc)

# Print reference averages
print("\nDifference-based metrics (exclude layer 0):")
print("Average Dice Coefficient:", np.mean(dice_list))
print("Average Jaccard Index:", np.mean(jaccard_list))
print("Average Pearson Correlation:", np.mean(pearson_list))
print("Average Spearman Correlation:", np.mean(spearman_list))
print("Average Cosine Similarity:", np.mean(cosine_list))
print("Average Symmetric KL Divergence:", np.mean(kl_list))
print("Average Wasserstein Distance:", np.mean(wasserstein_list))
print("Average Wasserstein Distance (Z-score):", np.mean(wasserstein_norm_list))
print("Average Euclidean Distance (Unit Norm):", np.mean(euclidean_list))

# -------------------------------------------------------------------
# 5) Plot difference-based metrics vs. layer index
# -------------------------------------------------------------------
plt.figure(figsize=(12, 10), dpi=300)

# If mean_diff_v3.shape[0] == 33, then layers are [0..32]
num_layers_diff = mean_diff_v3.shape[0]  
# layer_indices_diff = np.arange(num_layers_diff)
layer_indices_diff = np.arange(1, num_layers_diff+1)  # Will be [0, 1, 2, ..., 32]

metrics_dict = {
    f"Dice Coefficient (top{top_neurons})": (dice_list, "blue"),
    f"Jaccard Index (top{top_neurons})": (jaccard_list, "purple"),
    "Pearson Correlation": (pearson_list, "blue"),
    "Spearman Correlation": (spearman_list, "green"),
    "Cosine Similarity": (cosine_list, "red"),
    "KL Divergence": (kl_list, "brown"),
    "Wasserstein Distance": (wasserstein_list, "orange"),
    "Wasserstein Distance (Z-score)": (wasserstein_norm_list, "teal"),
    "Euclidean Distance (Unit Norm)": (euclidean_list, "cyan")
}

for i, (metric_name, (vals, color)) in enumerate(metrics_dict.items(), 1):
    plt.subplot(3, 3, i)
    plt.plot(layer_indices_diff, vals, marker='o', linestyle='-', color=color, linewidth=1.5)
    if i > 5:
        plt.xlabel("Layer Index", fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(metric_name, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.5)

# Add a large title for the whole figure
plt.suptitle("Similarity Metrics Across Layers Between Prompts v3 and v5", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to fit the title

diff_output_png = os.path.join(save_dir, "all_diff_metrics_vs_layers.png")
plt.savefig(diff_output_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"Difference-based metrics plot saved to: {diff_output_png}")

# -------------------------------------------------------------------
# 6) Dice coefficient as a function of top-N neurons
# -------------------------------------------------------------------
top_list = range(1, 100)
dice_avg_list, baseline_list = [], []
num_layers_used = mean_diff_v3.shape[0]  # same as num_layers_diff
hidden_size_used = mean_diff_v3.shape[1]

for top_val in top_list:
    dice_per_layer = []
    for layer_idx in range(num_layers_used):
        v3_layer = mean_diff_v3[layer_idx]
        v5_layer = mean_diff_v5[layer_idx]
        idx_v3 = set(np.argsort(np.abs(v3_layer))[-top_val:])
        idx_v5 = set(np.argsort(np.abs(v5_layer))[-top_val:])
        d_val = dice_coefficient(idx_v3, idx_v5)
        dice_per_layer.append(d_val)
    avg_dice = np.mean(dice_per_layer)
    dice_avg_list.append(avg_dice)
    baseline = top_val / hidden_size_used
    baseline_list.append(baseline)

plt.figure(figsize=(9, 6), dpi=300)
plt.plot(top_list, dice_avg_list, marker='o', markersize=4, linestyle='-', linewidth=1.5,
          color='#1f77b4', label='Observed Overlap')
plt.plot(top_list, baseline_list, linestyle='--', linewidth=1.2,
          color='#d62728', label='Random Baseline')

plt.xlabel("Number of Top Neurons", fontsize=12, fontweight='bold')
plt.ylabel("Mean Dice Coefficient", fontsize=12, fontweight='bold')
plt.title("Representational Overlap (v3 vs. v5, Excl. Layer 0)", fontsize=14, fontweight='bold')
plt.legend(frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
dice_output_png = os.path.join(save_dir, "representational_overlap_dice_coefficient.png")
plt.savefig(dice_output_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"Dice coefficient vs. top-N plot saved to {dice_output_png}")

# -------------------------------------------------------------------
# 7) Overlay similarity metrics on the raw hidden states (all layers),
#    comparing llama3_v3 vs. llama3_v5 with a common y-axis scale.
# -------------------------------------------------------------------
# Squeeze to shape: (num_layers, hidden_size)
expert_hs_v3 = np.squeeze(data_char_v3)       # shape: (num_layers, hidden_size)
none_expert_hs_v3 = np.squeeze(data_none_char_v3)
expert_hs_v5 = np.squeeze(data_char_v5)
none_expert_hs_v5 = np.squeeze(data_none_char_v5)

num_layers_all, hidden_size_all = expert_hs_v3.shape
print("\nRaw hidden states shape (including layer 0):")
print(f"llama3_v3: {expert_hs_v3.shape}, llama3_v5: {expert_hs_v5.shape}")

def compute_similarity(expert_hs, none_expert_hs):
    pearson_list, spearman_list, cosine_list = [], [], []
    eucl_list = []
    L = expert_hs.shape[0]
    for layer in range(L):
        expert_layer = expert_hs[layer]
        none_layer = none_expert_hs[layer]
        # Pearson & Spearman
        if np.std(expert_layer) < 1e-8 or np.std(none_layer) < 1e-8:
            p_val, s_val = np.nan, np.nan
        else:
            p_val = np.corrcoef(expert_layer, none_layer)[0, 1]
            s_val = stats.spearmanr(expert_layer, none_layer)[0]
        pearson_list.append(p_val)
        spearman_list.append(s_val)

        # Cosine similarity
        cos_val = cosine_similarity(expert_layer.reshape(1, -1), none_layer.reshape(1, -1))[0, 0]
        cosine_list.append(cos_val)

        # Euclidean Distance (Unit Vector Normalized)
        e_unit = expert_layer / np.linalg.norm(expert_layer)
        n_unit = none_layer / np.linalg.norm(none_layer)
        eucl_dist = np.linalg.norm(e_unit - n_unit)
        eucl_list.append(eucl_dist)

    return {
        "pearson": np.array(pearson_list),
        "spearman": np.array(spearman_list),
        "cosine": np.array(cosine_list),
        "euclidean": np.array(eucl_list)
    }

metrics_v3 = compute_similarity(expert_hs_v3, none_expert_hs_v3)
metrics_v5 = compute_similarity(expert_hs_v5, none_expert_hs_v5)

# Define metrics to plot (Wasserstein removed)
metric_names = ["pearson", "spearman", "cosine", "euclidean"]
ylabels = [
    "Pearson Correlation",
    "Spearman Correlation",
    "Cosine Similarity",
    "Euclidean Distance (Unit Norm)"
]

# Determine global y-limits across v3 & v5 for each metric
global_mins, global_maxs = {}, {}
for m in metric_names:
    global_mins[m] = min(np.nanmin(metrics_v3[m]), np.nanmin(metrics_v5[m]))
    global_maxs[m] = max(np.nanmax(metrics_v3[m]), np.nanmax(metrics_v5[m]))

# Plot overlay with same y-scale on a 2x2 grid
layers_all = np.arange(0, num_layers_all)  # Layers labeled 0 to num_layers-1
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

for i, m in enumerate(metric_names):
    ax = axes[i]
    ax.plot(layers_all, metrics_v3[m], marker='o', color='blue', linestyle='-', label='v3')
    ax.plot(layers_all, metrics_v5[m], marker='s', color='red', linestyle='--', label='v5')
    if i >1:
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabels[i], fontsize=12, fontweight='bold')
    ax.set_title(ylabels[i], fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_ylim(global_mins[m], global_maxs[m])

fig.suptitle("Similarity between Hidden States of Expert and None Expert\n(prompt v3 vs. prompt v5)", 
              fontsize=16, fontweight='bold')
# plt.tight_layout(rect=[0, 0, 1, 0.95])
final_output = os.path.join(save_dir, "similarity_comparison_v3_vs_v5.png")
plt.savefig(final_output, dpi=300, bbox_inches="tight")
plt.show()
print("Final comparison plot saved to:", final_output)