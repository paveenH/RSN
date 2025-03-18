#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads the npz file containing the mean HS-diff matrix for each answer pair,
and prints the keys, shapes, and basic statistics (min, max, mean) of each matrix.

Usage example:
    python read_pairs_matrix.py /path/to/all_pairs_mean_1B.npz
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity  

# -----------------------------------------------------------
# Part 1: Plot Answer Pairs Distribution
# -----------------------------------------------------------
# Data from the provided list
pairs = [
    ("A-A", 1591), ("A-B", 75), ("A-D", 78), ("A-E", 735), ("A-C", 69),
    ("B-A", 29), ("B-B", 2537), ("B-D", 106), ("B-E", 825), ("B-C", 50),
    ("D-A", 39), ("D-B", 29), ("D-D", 2586), ("D-E", 660), ("D-C", 34),
    ("E-A", 0), ("E-B", 3), ("E-D", 4), ("E-E", 360), ("E-C", 2),
    ("C-A", 47), ("C-B", 65), ("C-D", 126), ("C-E", 1118), ("C-C", 2807)
]

# Sort pairs by sample count in descending order
pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

# Separate sorted pairs and counts
pair_labels, sample_counts = zip(*pairs_sorted)

# Plot distribution
plt.figure(figsize=(12, 6))
plt.bar(pair_labels, sample_counts, color="royalblue")

plt.xlabel("Pair Type", fontsize=12, fontweight='bold')
plt.ylabel("Sample Count", fontsize=12, fontweight='bold')
plt.title("Distribution of Answer Pairs", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()

# -----------------------------------------------------------
# Part 2: Plot Answer Pairs Distribution
# -----------------------------------------------------------

def dice_coefficient(set1, set2):
    """Dice coefficient: 2 * intersection / (|set1| + |set2|)."""
    intersection = len(set1.intersection(set2))
    if (len(set1) + len(set2)) == 0:
        return np.nan
    return 2.0 * intersection / (len(set1) + len(set2))

def jaccard_index(set1, set2):
    """Jaccard index: intersection / union."""
    union_size = len(set1.union(set2))
    if union_size == 0:
        return np.nan
    return len(set1.intersection(set2)) / union_size

def symmetric_kl(v1, v2):
    """Symmetric KL divergence: normalize absolute vectors using softmax."""
    exp_v1 = np.exp(np.abs(v1))
    exp_v2 = np.exp(np.abs(v2))
    p = exp_v1 / np.sum(exp_v1)
    q = exp_v2 / np.sum(exp_v2)
    kl1 = stats.entropy(p, q)
    kl2 = stats.entropy(q, p)
    return (kl1 + kl2) / 2.0

def wasserstein_distance_zscore(v1, v2):
    """
    Normalize vectors using Z-score, then compute Wasserstein distance.
    Optionally, you can return the unnormalized distance depending on the need.
    """
    m1, s1 = np.mean(v1), np.std(v1)
    m2, s2 = np.mean(v2), np.std(v2)
    if s1 == 0: 
        v1_norm = v1 - m1
    else:
        v1_norm = (v1 - m1) / s1
    if s2 == 0:
        v2_norm = v2 - m2
    else:
        v2_norm = (v2 - m2) / s2
    return stats.wasserstein_distance(v1_norm, v2_norm)

def euclidean_distance_unit_norm(v1, v2):
    """Normalize to unit vectors, then compute Euclidean distance."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        # If a zero vector appears, do not normalize
        return np.linalg.norm(v1 - v2)
    else:
        v1_unit = v1 / norm1
        v2_unit = v2 / norm2
        return np.linalg.norm(v1_unit - v2_unit)

def compute_layer_metrics(v1, v2, top_n=20):
    """
    Calculate various metrics between two layer vectors:
      1) Dice coefficient (top-N)
      2) Jaccard index (top-N)
      3) Pearson correlation
      4) Spearman correlation
      5) Cosine similarity
      6) Symmetric KL divergence
      7) Wasserstein distance (Z-score)
      8) Euclidean distance (unit norm)
    """
    # 1) Dice & Jaccard
    top_v1 = set(np.argsort(np.abs(v1))[-top_n:])
    top_v2 = set(np.argsort(np.abs(v2))[-top_n:])
    dice_val = dice_coefficient(top_v1, top_v2)
    jaccard_val = jaccard_index(top_v1, top_v2)

    # 2) Pearson & Spearman
    if np.std(v1) < 1e-8 or np.std(v2) < 1e-8:
        pearson_val = np.nan
        spearman_val = np.nan
    else:
        pearson_val = np.corrcoef(v1, v2)[0, 1]
        spearman_val = stats.spearmanr(v1, v2)[0]
    
    # 3) Cosine similarity
    cos_val = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
    
    # 4) Distance
    kl_val = symmetric_kl(v1, v2)
    wdist_val = wasserstein_distance_zscore(v1, v2)
    euc_val = euclidean_distance_unit_norm(v1, v2)
     
    return dice_val, jaccard_val, pearson_val, spearman_val, cos_val, kl_val, wdist_val, euc_val

path = os.getcwd()
model = "llama3_v3_pair"  
size = "8B"
top_n = 20

data_path = os.path.join(path, model)
save_path = os.path.join(path, "plot", model)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
npz_file = os.path.join(data_path, "all_pairs_mean_8B.npz")
data = np.load(npz_file)
keys = sorted(list(data.keys()))

print("Loaded keys:", keys)
print(f"There are {len(keys)} keys.")

# Reshape data for each key to (L, H)
matrices = {}
for key in keys:
    arr = data[key]  # Possibly shape=(1, L, H) or other
    # Assuming shape is (1, L, H), use np.squeeze to remove axis=0
    m = np.squeeze(arr, axis=0)  
    matrices[key] = m  # shape=(L, H)

sample_key = keys[0]
L, H = matrices[sample_key].shape
print(f"Example key={sample_key} matrix shape: ({L} layers, {H} hidden units)")

# ---------------------------
# 3) Calculate pairwise metrics and get 9 large metrics matrices
# ---------------------------
n_keys = len(keys)

# Prepare nine (n_keys x n_keys) matrices
dice_matrix         = np.zeros((n_keys, n_keys))
jaccard_matrix      = np.zeros((n_keys, n_keys))
pearson_matrix      = np.zeros((n_keys, n_keys))
spearman_matrix     = np.zeros((n_keys, n_keys))
cosine_matrix       = np.zeros((n_keys, n_keys))
kl_matrix           = np.zeros((n_keys, n_keys))
wasserstein_matrix  = np.zeros((n_keys, n_keys))
euclidean_matrix    = np.zeros((n_keys, n_keys))

# Calculate for each key pair
for i, key1 in enumerate(keys):
    m1 = matrices[key1]  # shape=(L, H)
    for j, key2 in enumerate(keys):
        m2 = matrices[key2]
        
        # Calculate metrics per layer and average them
        dice_list, jaccard_list = [], []
        pearson_list, spearman_list, cos_list = [], [], []
        kl_list, wdist_list, euc_list = [], [], []
        
        for layer_idx in range(L):
            v1 = m1[layer_idx]
            v2 = m2[layer_idx]
            (d_val, j_val, p_val, s_val, c_val, kl_val, w_val, e_val) = compute_layer_metrics(v1, v2, top_n)
            
            dice_list.append(d_val)
            jaccard_list.append(j_val)
            pearson_list.append(p_val)
            spearman_list.append(s_val)
            cos_list.append(c_val)
            kl_list.append(kl_val)
            wdist_list.append(w_val)
            euc_list.append(e_val)
            
        
        dice_matrix[i, j]        = np.mean(dice_list)
        jaccard_matrix[i, j]     = np.mean(jaccard_list)
        pearson_matrix[i, j]     = np.mean(pearson_list)
        spearman_matrix[i, j]    = np.mean(spearman_list)
        cosine_matrix[i, j]      = np.mean(cos_list)
        kl_matrix[i, j]          = np.mean(kl_list)
        wasserstein_matrix[i, j] = np.mean(wdist_list)
        euclidean_matrix[i, j]   = np.mean(euc_list)

# ---------------------------
# 4) Print the global averages
# ---------------------------
print("\n===== Average Metrics (Averaged Across All Key-Pairs)=====")
print(f"Average Dice Coefficient (top-{top_n}): {dice_matrix.mean():.6f}")
print(f"Average Jaccard Index (top-{top_n}): {jaccard_matrix.mean():.6f}")
print(f"Average Pearson Correlation: {np.nanmean(pearson_matrix):.6f}")
print(f"Average Spearman Correlation: {np.nanmean(spearman_matrix):.6f}")
print(f"Average Cosine Similarity: {cosine_matrix.mean():.6f}")
print(f"Average Symmetric KL Divergence: {kl_matrix.mean():.6f}")
print(f"Average Wasserstein Distance (Z-score): {wasserstein_matrix.mean():.6f}")
print(f"Average Euclidean Distance (Unit Norm): {euclidean_matrix.mean():.6f}")

# ---------------------------
# 5) Visualization: Create a 24x24 heatmap for each metric (or n_keys x n_keys)
# ---------------------------
# If you want to plot multiple subplots at once, you can wrap the code. Here, we'll show each one individually.
metric_dict = {
    f"Dice (top {top_n})": dice_matrix,
    f"Jaccard (top {top_n})": jaccard_matrix,
    "Pearson Correlation": pearson_matrix,
    "Spearman Correlation": spearman_matrix,
    "Cosine Similarity": cosine_matrix,
    "Symmetric KL Divergence": kl_matrix,
    "Wasserstein Distance (Z-score)": wasserstein_matrix,
    "Euclidean Distance (Unit Norm)": euclidean_matrix
}

for metric_name, mat in metric_dict.items():
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(mat, annot=False, cmap="viridis",
                xticklabels=keys, yticklabels=keys)
    plt.title(metric_name, fontsize=14, fontweight='bold')
    plt.xlabel("Answer Pair")
    plt.ylabel("Answer Pair")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_png = os.path.join(save_path, f"{metric_name.replace(' ', '_')}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Heatmap saved to: {out_png}")