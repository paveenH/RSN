#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare similarity metrics across layers between:
1) Expert group: v3 vs. v5 (expert vs. none_expert)
2) Student group: beginner vs. advanced

Created on Thu Mar  6 17:58:32 2025

@author: paveen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# 1) Global parameters
# ==============================
sns.set_theme(style="white", context="paper", font_scale=1.2)
path = os.getcwd()
size = "8B"

# ============ Expert data (v3 & v5) ============
model_v3 = "llama3_v3"
model_v5 = "llama3_v5"
data_path_v3 = os.path.join(path, model_v3)
data_path_v5 = os.path.join(path, model_v5)

file_char_v3 = os.path.join(data_path_v3, f"all_mean_{size}.npy")       # v3:  expert
file_none_char_v3 = os.path.join(data_path_v3, f"none_all_mean_{size}.npy")  # v3:  none_expert
file_char_v5 = os.path.join(data_path_v5, f"all_mean_{size}.npy")       # v5:  expert
file_none_char_v5 = os.path.join(data_path_v5, f"none_all_mean_{size}.npy")  # v5:  none_expert

# ============ Student data (beginner & advanced) ============
model_stu = "llama3_v3_stu"
data_path_stu = os.path.join(path, model_stu)
file_beginner = os.path.join(data_path_stu, f"beginner_all_mean_{size}.npy")
file_advanced = os.path.join(data_path_stu, f"advanced_all_mean_{size}.npy")

# ============ Save directory ============
save_dir = os.path.join(path, "plot", "comparison_expert_and_student")
os.makedirs(save_dir, exist_ok=True)

# ==============================
# 2) Load  data
# ==============================
data_char_v3 = np.load(file_char_v3)         # expert (v3)
data_none_char_v3 = np.load(file_none_char_v3)  # none_expert (v3)
data_char_v5 = np.load(file_char_v5)         # expert (v5)
data_none_char_v5 = np.load(file_none_char_v5)  # none_expert (v5)

print("Loaded Expert shapes:")
print("v3 expert:", data_char_v3.shape, "v3 none_expert:", data_none_char_v3.shape)
print("v5 expert:", data_char_v5.shape, "v5 none_expert:", data_none_char_v5.shape)

data_beginner = np.load(file_beginner)   # student: beginner
data_advanced = np.load(file_advanced)   # student: advanced
print("Loaded Student shapes:")
print("beginner:", data_beginner.shape, " advanced:", data_advanced.shape)

# ==============================
# 3)Similarity-computing
# ==============================
def compute_similarity(arr1, arr2):
    """
    Compute similarity metrics between arr1 and arr2, each shape = (num_layers, hidden_size).
    Return dict of arrays: {pearson, spearman, cosine, euclidean}
    """
    num_layers = arr1.shape[0]
    pearson_list, spearman_list, cosine_list = [], [], []
    euclidean_list = []

    for layer in range(num_layers):
        v1 = arr1[layer]
        v2 = arr2[layer]
        # Pearson & Spearman
        if np.std(v1) < 1e-8 or np.std(v2) < 1e-8:
            p_val = np.nan
            s_val = np.nan
        else:
            p_val = np.corrcoef(v1, v2)[0, 1]
            s_val = stats.spearmanr(v1, v2)[0]
        pearson_list.append(p_val)
        spearman_list.append(s_val)

        # Cosine
        cos_val = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
        cosine_list.append(cos_val)

        # Euclidean (unit norm)
        if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
            euc = np.nan
        else:
            v1_unit = v1 / np.linalg.norm(v1)
            v2_unit = v2 / np.linalg.norm(v2)
            euc = np.linalg.norm(v1_unit - v2_unit)
        euclidean_list.append(euc)

    return {
        "pearson": np.array(pearson_list),
        "spearman": np.array(spearman_list),
        "cosine": np.array(cosine_list),
        "euclidean": np.array(euclidean_list),
    }


# Squeeze to shape: (num_layers, hidden_size)
expert_v3 = np.squeeze(data_char_v3)        # v3: expert
none_v3   = np.squeeze(data_none_char_v3)   # v3: none_expert
expert_v5 = np.squeeze(data_char_v5)        # v5: expert
none_v5   = np.squeeze(data_none_char_v5)   # v5: none_expert

num_layers_expert, hidden_size_expert = expert_v3.shape
print("\nExpert raw hidden states shape:")
print(f"v3: {expert_v3.shape}, v5: {expert_v5.shape}")

# Already shape: (num_layers, hidden_size)
beginner_hs = np.squeeze(data_beginner)    # student: beginner
advanced_hs = np.squeeze(data_advanced)    # student: advanced
num_layers_student, hidden_size_student = beginner_hs.shape
print("\nStudent raw hidden states shape:")
print(f"beginner: {beginner_hs.shape}, advanced: {advanced_hs.shape}")

# Expert: v3_expert vs. v3_none_expert
metrics_expert_v3 = compute_similarity(expert_v3, none_v3)
# Expert: v5_expert vs. v5_none_expert
metrics_expert_v5 = compute_similarity(expert_v5, none_v5)
# Student: beginner vs. advanced
metrics_student = compute_similarity(beginner_hs, advanced_hs)

# ==============================
# 4) Plot similarity
# ==============================
metric_names = ["pearson", "spearman", "cosine", "euclidean"]
ylabels = [
    "Pearson Correlation",
    "Spearman Correlation",
    "Cosine Similarity",
    "Euclidean Distance (Unit Norm)"
]

global_mins, global_maxs = {}, {}
for m in metric_names:
    # Collect all data from three comparisons
    arr_v3 = metrics_expert_v3[m]
    arr_v5 = metrics_expert_v5[m]
    arr_stu= metrics_student[m]
    # If all are NaN, set 0..1 as default, else find min..max
    all_data = np.concatenate([arr_v3, arr_v5, arr_stu])
    if np.all(np.isnan(all_data)):
        global_mins[m], global_maxs[m] = 0.0, 1.0
    else:
        valid_data = all_data[~np.isnan(all_data)]
        if len(valid_data) == 0:
            global_mins[m], global_maxs[m] = 0.0, 1.0
        else:
            global_mins[m] = np.min(valid_data)
            global_maxs[m] = np.max(valid_data)


fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

layers_expert = np.arange(num_layers_expert)   # 0..(num_layers_expert-1)
layers_student= np.arange(num_layers_student)  # 0..(num_layers_student-1)

for i, m in enumerate(metric_names):
    ax = axes[i]
    # Expert v3 (blue solid)
    ax.plot(layers_expert, metrics_expert_v3[m], marker='o', color='blue', linestyle='-',
            label='Expert (v3)') 
    # Expert v5 (red dashed)
    ax.plot(layers_expert, metrics_expert_v5[m], marker='s', color='red', linestyle='--',
            label='Expert (v5)')
    # Student (green dotted)
    ax.plot(layers_student, metrics_student[m], marker='^', color='green', linestyle=':',
            label='student (v3')
    
    if i > 1:
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabels[i], fontsize=12, fontweight='bold')
    ax.set_title(ylabels[i], fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Set global y-limits
    ax.set_ylim(global_mins[m], global_maxs[m])

fig.suptitle("Similarity Metrics Across Layers",
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])

final_output = os.path.join(save_dir, "similarity_comparison_expert_vs_student.png")
plt.savefig(final_output, dpi=300, bbox_inches="tight")
plt.show()
print("Final comparison plot saved to:", final_output)

# =============================
# 5) Compute difference-based data
# =============================
# Expert diff v3 / none_v3, v5 / none_v5 (exclude layer 0)
data_char_diff_v3 = data_char_v3 - data_none_char_v3
data_char_diff_v5 = data_char_v5 - data_none_char_v5
char_diff_v3 = data_char_diff_v3[:, :, 1:, :]  # exclude layer 0
char_diff_v5 = data_char_diff_v5[:, :, 1:, :]

# Student diff = advanced - beginner (exclude layer 0 as well)
data_diff_stu = data_advanced - data_beginner
char_diff_stu = data_diff_stu[:, :, 1:, :]

mean_diff_v3 = np.mean(char_diff_v3, axis=(0, 1))   # shape: (layers-1, hidden_size)
mean_diff_v5 = np.mean(char_diff_v5, axis=(0, 1))
mean_diff_stu= np.mean(char_diff_stu, axis=(0, 1))

num_layers_diff, hidden_size_diff = mean_diff_v3.shape
print(f"\nAfter exclude layer 0: {num_layers_diff} layers remain.")
top_neurons = 20

def dice_coefficient(set1, set2):
    intersection = len(set1.intersection(set2))
    denom = (len(set1) + len(set2))
    return 2 * intersection / denom if denom > 0 else np.nan

def compute_diff_metrics(mean_diff1, mean_diff2, topN=20):
    dice_list, jaccard_list = [], []
    pearson_list, spearman_list, cosine_list = [], [], []
    kl_list, wass_list, wass_norm_list, eucl_list = [], [], [], []

    num_layers_now = mean_diff1.shape[0]
    for layer_idx in range(num_layers_now):
        arr1 = mean_diff1[layer_idx]
        arr2 = mean_diff2[layer_idx]

        # Top-N neurons
        top1 = set(np.argsort(np.abs(arr1))[-topN:])
        top2 = set(np.argsort(np.abs(arr2))[-topN:])

        # Dice & Jaccard
        d_val = dice_coefficient(top1, top2)
        union_set = top1.union(top2)
        intersection_set = top1.intersection(top2)
        j_val = len(intersection_set)/len(union_set) if len(union_set)>0 else np.nan
        dice_list.append(d_val)
        jaccard_list.append(j_val)

        # Pearson & Spearman
        p_val = np.corrcoef(arr1, arr2)[0,1]
        s_val = stats.spearmanr(arr1, arr2)[0]
        pearson_list.append(p_val)
        spearman_list.append(s_val)

        # Cosine
        cos_val = cosine_similarity(arr1.reshape(1, -1), arr2.reshape(1, -1))[0,0]
        cosine_list.append(cos_val)

        # KL (softmax on abs)
        abs1 = np.exp(np.abs(arr1)); abs1 /= abs1.sum()
        abs2 = np.exp(np.abs(arr2)); abs2 /= abs2.sum()
        kl_sym = (stats.entropy(abs1, abs2) + stats.entropy(abs2, abs1))/2
        kl_list.append(kl_sym)

        # Wasserstein (unnormalized)
        w_dist = stats.wasserstein_distance(arr1, arr2)
        wass_list.append(w_dist)

        # Wasserstein (z-score)
        m1, s1 = arr1.mean(), arr1.std()
        m2, s2 = arr2.mean(), arr2.std()
        arr1_norm = (arr1 - m1)/s1 if s1>0 else (arr1 - m1)
        arr2_norm = (arr2 - m2)/s2 if s2>0 else (arr2 - m2)
        w_norm = stats.wasserstein_distance(arr1_norm, arr2_norm)
        wass_norm_list.append(w_norm)

        # Euclidean (unit norm)
        if np.linalg.norm(arr1)<1e-12 or np.linalg.norm(arr2)<1e-12:
            e_val = np.nan
        else:
            arr1_unit = arr1/np.linalg.norm(arr1)
            arr2_unit = arr2/np.linalg.norm(arr2)
            e_val = np.linalg.norm(arr1_unit - arr2_unit)
        eucl_list.append(e_val)

    return {
        "dice": dice_list,
        "jaccard": jaccard_list,
        "pearson": pearson_list,
        "spearman": spearman_list,
        "cosine": cosine_list,
        "kl": kl_list,
        "wass": wass_list,
        "wass_norm": wass_norm_list,
        "eucl": eucl_list
    }

# Expert v3 vs. v5
metrics_v3_v5 = compute_diff_metrics(mean_diff_v3, mean_diff_v5, topN=top_neurons)
# Student vs. Expert v3
metrics_stu_v3 = compute_diff_metrics(mean_diff_stu, mean_diff_v3, topN=top_neurons)

print("\nDifference-based metrics (exclude layer 0):")
def print_averages(msg, dic):
    print(msg)
    print("Average Dice Coefficient:", np.mean(dic["dice"]))
    print("Average Jaccard Index:", np.mean(dic["jaccard"]))
    print("Average Pearson Correlation:", np.mean(dic["pearson"]))
    print("Average Spearman Correlation:", np.mean(dic["spearman"]))
    print("Average Cosine Similarity:", np.mean(dic["cosine"]))
    print("Average Symmetric KL Divergence:", np.mean(dic["kl"]))
    print("Average Wasserstein Distance:", np.mean(dic["wass"]))
    print("Average Wasserstein Distance (Z-score):", np.mean(dic["wass_norm"]))
    print("Average Euclidean Distance (Unit Norm):", np.mean(dic["eucl"]))

print_averages("**v3 vs. v5**:", metrics_v3_v5)
print_averages("**Student (advanced-beginner) vs. Expert v3**:", metrics_stu_v3)

# =============================
# 6) Plot difference-based metrics vs. layer index
# =============================
# plt.figure(figsize=(12, 10), dpi=300)
plt.figure(figsize=(14, 12), dpi=300)

layer_indices_diff = np.arange(1, num_layers_diff+1)  # If shape[0]==33, then layers= [1..33]

metrics_dict_exp = {
    f"Dice (top{top_neurons})": (metrics_v3_v5["dice"], "blue"),
    f"Jaccard (top{top_neurons})": (metrics_v3_v5["jaccard"], "purple"),
    "Pearson Corr": (metrics_v3_v5["pearson"], "blue"),
    "Spearman Corr": (metrics_v3_v5["spearman"], "green"),
    "Cosine": (metrics_v3_v5["cosine"], "red"),
    # "KL Div": (metrics_v3_v5["kl"], "brown"),
    # "Wasserstein": (metrics_v3_v5["wass"], "orange"),
    # "Wasserstein (Z)": (metrics_v3_v5["wass_norm"], "teal"),
    "Euclidean (Unit)": (metrics_v3_v5["eucl"], "cyan")
}

metrics_dict_stu = {
    "Dice (stu vs. v3)": metrics_stu_v3["dice"],
    "Jacc (stu vs. v3)": metrics_stu_v3["jaccard"],
    "Pearson(stu)": metrics_stu_v3["pearson"],
    "Spearman(stu)": metrics_stu_v3["spearman"],
    "Cosine(stu)": metrics_stu_v3["cosine"],
    # "KL(stu)": metrics_stu_v3["kl"],
    # "Wass(stu)": metrics_stu_v3["wass"],
    # "WassZ(stu)": metrics_stu_v3["wass_norm"],
    "Eucl(stu)": metrics_stu_v3["eucl"],
}

for i, (metric_name, (vals, color)) in enumerate(metrics_dict_exp.items(), 1):
    plt.subplot(3, 3, i)
    plt.plot(layer_indices_diff, vals, marker='o', linestyle='-', color=color, linewidth=1.5, label='v3 vs. v5')
    # Plot student vs. v3 in the same subplot (with different marker/style)
    # match same metric in metrics_dict_stu by index
    if i==1:   st_name="Dice (stu vs. v3)"; st_color='black'
    elif i==2: st_name="Jacc (stu vs. v3)"; st_color='gray'
    elif i==3: st_name="Pearson(stu)";      st_color='black'
    elif i==4: st_name="Spearman(stu)";     st_color='gray'
    elif i==5: st_name="Cosine(stu)";       st_color='black'
    # elif i==6: st_name="KL(stu)";           st_color='gray'
    # elif i==7: st_name="Wass(stu)";         st_color='black'
    # elif i==8: st_name="WassZ(stu)";        st_color='gray'
    elif i==6: st_name="Eucl(stu)";         st_color='black'
    st_vals = metrics_dict_stu[st_name]
    plt.plot(layer_indices_diff, st_vals, marker='s', linestyle='--', color=st_color, linewidth=1.2, label='Student vs. v3')

    if i>2:
        plt.xlabel("Layer Index", fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(metric_name, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

plt.suptitle("Difference-Based Metrics Across Layers\nExpert: v3 vs. v5,  Student: adv vs. beg vs. v3", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])
diff_output_png = os.path.join(save_dir, "all_diff_metrics_vs_layers_with_student.png")
plt.savefig(diff_output_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"Difference-based metrics plot saved to: {diff_output_png}")