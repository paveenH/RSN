#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:45:22 2025

@author: paveenhuang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity

path = os.getcwd()
model = "llama3_v5"  # llama3_v3
size = "8B"
data_path = path + f"/{model}"
save = path + f"/plot/{model}"
if not os.path.exists(save):
    os.mkdir(save)

task = "all_mean"
task_name = task.replace("_", " ")

data_char = np.load(f"{data_path}/{task}_{size}.npy")
data_none_char = np.load(f"{data_path}/none_{task}_{size}.npy")

num_samples = data_char.shape[0]
num_time = data_char.shape[1]
num_layers = data_char.shape[2]
hidden_size = data_char.shape[3]

print("char shape:", data_char.shape)
print("none char shape:", data_none_char.shape)
print(f"Data loaded successfully. Plots will be saved in: {save}")

char_differences = data_char - data_none_char
char_differences = char_differences[:, :, :, :]
print("differences shape:", char_differences.shape)


# ---------------------------
# 1) Max and Mean Value
# ---------------------------
start = 0
end = num_layers  # Use all layers: 0 to num_layers-1

# Max and Min samples and layers
max_char = data_char[:, :, start:end, :].max(axis=(0, 3))  # Shape: (num_layers,)
min_char = data_char[:, :, start:end, :].min(axis=(0, 3))  # Shape: (num_layers,)
max_none_char = data_none_char[:, :, start:end, :].max(axis=(0, 3))  # Shape: (num_layers,)
min_none_char = data_none_char[:, :, start:end, :].min(axis=(0, 3))  # Shape: (num_layers,)

layers = np.arange(start, end)  # This creates an array [0, 1, 2, ..., num_layers-1]

# Create a single figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot char max and min
ax.plot(layers, max_char.squeeze(), label="Char Max", color="blue", linestyle="-", marker="o", linewidth=2)
ax.plot(layers, min_char.squeeze(), label="Char Min", color="blue", linestyle="--", marker="x", linewidth=2)


# Plot none-char max and min
ax.plot(layers, max_none_char.squeeze(), label="None Char Max", color="orange", linestyle="-", marker="o", linewidth=2)
ax.plot(layers, min_none_char.squeeze(), label="None Char Min", color="orange", linestyle="--", marker="x", linewidth=2)

# Customize the plot
ax.set_title(f"Max and Min Activations at Different Layers ({model[:-3]} {size})", fontsize=18, fontweight="bold")
ax.set_xlabel("Layers", fontsize=14, fontweight="bold")
ax.set_ylabel("Activation Values", fontsize=14, fontweight="bold")
ax.tick_params(axis="both", which="major", labelsize=12, width=1.5)
ax.legend(fontsize=12, frameon=True)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the title
save_path = os.path.join(save, f"max_min_combined_activations_{task}_{size}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Image has been saved to {save_path}")

plt.show()

# # ---------------------------
# # 2) Mean Difference Without Order
# # ---------------------------
# mean_diff = char_differences.mean(axis=0).squeeze()  # Shape: (layers, neurons)
# sorted_diff = np.sort(mean_diff, axis=1)  # Sort each row (layer) by neuron values
# abs_max = max(abs(np.min(sorted_diff)), abs(np.max(sorted_diff)))

# fig, ax = plt.subplots(figsize=(10, 6))
# im = ax.imshow(sorted_diff, aspect="auto", cmap="seismic", origin="lower", vmin=-abs_max, vmax=abs_max)

# # Set y-axis ticks: label rows as layers 0 to 32 (assuming 33 layers)
# num_layers = sorted_diff.shape[0]
# ax.set_yticks(np.arange(num_layers))
# ax.set_yticklabels(np.arange(num_layers), fontsize=12)

# # Add labels and title with paper-style formatting
# ax.set_title(f"Sorted Neuron-Wise Mean Difference ({model[:-3]} {size})", fontsize=16, fontweight="bold")
# ax.set_xlabel("Neuron (sorted by mean difference)", fontsize=14, fontweight="bold")
# ax.set_ylabel("Layer", fontsize=14, fontweight="bold")

# # Add colorbar with label styling
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label("Mean Difference", fontsize=14, fontweight="bold")

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# save_path = os.path.join(save, f"sorted_neuron_mean_difference_{model}_{size}.png")
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
# print(f"Image saved to {save_path}")
# plt.show()

# # ---------------------------
# # 3) Similarity
# # ---------------------------
# # Squeeze to shape: (33, 4096) where 33 = number of layers
# expert_hs = np.squeeze(data_char)  # expert hidden states
# none_expert_hs = np.squeeze(data_none_char)  # none-expert hidden states

# num_layers, hidden_size = expert_hs.shape
# # Prepare lists to store similarity metrics for each layer
# # Prepare lists to store similarity metrics for each layer
# pearson_list, spearman_list, cosine_list = [], [], []
# wasserstein_list, euclidean_list = [], []

# for layer in range(num_layers):
#     expert_layer = expert_hs[layer]
#     none_expert_layer = none_expert_hs[layer]

#     # Compute Pearson and Spearman correlations
#     if np.std(expert_layer) < 1e-8 or np.std(none_expert_layer) < 1e-8:
#         pearson = np.nan
#         spearman = np.nan
#     else:
#         pearson = np.corrcoef(expert_layer, none_expert_layer)[0, 1]
#         spearman = stats.spearmanr(expert_layer, none_expert_layer)[0]
#     pearson_list.append(pearson)
#     spearman_list.append(spearman)

#     # Compute Cosine similarity
#     cos_sim = cosine_similarity(expert_layer.reshape(1, -1), none_expert_layer.reshape(1, -1))[0, 0]
#     cosine_list.append(cos_sim)

#     # Compute Wasserstein Distance (Z-score normalized)
#     expert_mean, expert_std = np.mean(expert_layer), np.std(expert_layer)
#     none_expert_mean, none_expert_std = np.mean(none_expert_layer), np.std(none_expert_layer)
#     expert_norm = (expert_layer - expert_mean) / expert_std if expert_std > 0 else (expert_layer - expert_mean)
#     none_expert_norm = (
#         (none_expert_layer - none_expert_mean) / none_expert_std
#         if none_expert_std > 0
#         else (none_expert_layer - none_expert_mean)
#     )
#     wasserstein_distance = stats.wasserstein_distance(expert_norm, none_expert_norm)
#     wasserstein_list.append(wasserstein_distance)

#     # Compute Euclidean Distance (Unit Vector Normalized)
#     expert_unit = expert_layer / np.linalg.norm(expert_layer)
#     none_expert_unit = none_expert_layer / np.linalg.norm(none_expert_layer)
#     euclidean_distance = np.linalg.norm(expert_unit - none_expert_unit)
#     euclidean_list.append(euclidean_distance)

# # Print average similarity metrics across layers
# print("Average Pearson Correlation:", np.nanmean(pearson_list))
# print("Average Spearman Correlation:", np.nanmean(spearman_list))
# print("Average Cosine Similarity:", np.mean(cosine_list))
# print("Average Wasserstein Distance (Z-score):", np.mean(wasserstein_list))
# print("Average Euclidean Distance (Unit Norm):", np.mean(euclidean_list))

# # Plot each similarity metric vs. layer index in separate figures
# layers = np.arange(0, num_layers)

# # Create a 2x2 grid for subplots
# fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# axes = axes.flatten()

# # Add a large title for the entire figure
# fig.suptitle(f"Similarity between Hidden States of Expert and None Expert ({model[:-3]}-{size})", fontsize=16, fontweight="bold")
# plt.subplots_adjust(top=0.90)  # Adjust top to make room for the suptitle

# # Plot Pearson Correlation
# axes[0].plot(layers, pearson_list, marker="o", color="blue", label="Pearson")
# axes[0].set_ylabel("Pearson Correlation", fontsize=12, fontweight="bold")
# axes[0].set_title("Pearson Correlation", fontsize=14, fontweight="bold")
# axes[0].grid(True, linestyle="--", alpha=0.5)

# # Plot Spearman Correlation
# axes[1].plot(layers, spearman_list, marker="s", color="green", label="Spearman")
# axes[1].set_ylabel("Spearman Correlation", fontsize=12, fontweight="bold")
# axes[1].set_title("Spearman Correlation", fontsize=14, fontweight="bold")
# axes[1].grid(True, linestyle="--", alpha=0.5)

# # Plot Cosine Similarity
# axes[2].plot(layers, cosine_list, marker="^", color="red", label="Cosine")
# axes[2].set_xlabel("Layer Index", fontsize=12, fontweight="bold")
# axes[2].set_ylabel("Cosine Similarity", fontsize=12, fontweight="bold")
# axes[2].set_title("Cosine Similarity", fontsize=14, fontweight="bold")
# axes[2].grid(True, linestyle="--", alpha=0.5)

# # Plot Euclidean Distance (Unit Norm)
# axes[3].plot(layers, euclidean_list, marker="d", color="cyan", label="Euclidean (Unit Norm)")
# axes[3].set_xlabel("Layer Index", fontsize=12, fontweight="bold")
# axes[3].set_ylabel("Euclidean Distance (Unit Norm)", fontsize=12, fontweight="bold")
# axes[3].set_title("Euclidean Distance (Unit Norm)", fontsize=14, fontweight="bold")
# axes[3].grid(True, linestyle="--", alpha=0.5)

# # plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.tight_layout()
# output_file = os.path.join(save, "all_similarity_metrics.png")
# plt.savefig(output_file, dpi=300, bbox_inches="tight")
# plt.show()
# print("Combined plot saved to:", output_file)


# -----------------------------
# 4) Plot top-N neurons scatter
# ------------------------------
# Step 1: Compute mean neuron-wise mean difference (NMD) for each neuron across samples
mean_diff = char_differences.squeeze(0).squeeze(0)   # Shape: (layers, neurons)

# Use top = 20 neurons per layer
top = 20
print(f"Plot the top {top} neurons per layer.")

# Step 2: Identify top neurons for each layer
top_neurons_per_layer = []
top_indices_matrix = np.zeros((mean_diff.shape[0], top), dtype=int)
top_values_matrix = np.zeros((mean_diff.shape[0], top))

for layer_idx in range(mean_diff.shape[0]):
    layer_values = mean_diff[layer_idx]  # NMD values for current layer
    top_indices = np.argsort(np.abs(layer_values))[-top:]  # Indices of top neurons (by absolute value)
    top_indices_matrix[layer_idx] = top_indices
    top_values_matrix[layer_idx] = layer_values[top_indices]
    top_neurons_per_layer.append((layer_idx, top_indices))

# # Step 3: Prepare scatter plot data
# layer_positions = []
# neuron_indices = []
# top_values = []

# for layer_idx, top_indices in top_neurons_per_layer:
#     # Convert layer index to 1-based numbering for plotting
#     layer_positions.extend([layer_idx + 1] * len(top_indices))
#     neuron_indices.extend(top_indices)
#     top_values.extend(mean_diff[layer_idx, top_indices])

# # Calculate symmetric color limits for colorbar
# abs_max = max(abs(np.min(top_values)), abs(np.max(top_values)))

# # Step 4: Create scatter plot with publication-quality styling
# fig, ax = plt.subplots(figsize=(12, 6))
# scatter = ax.scatter(
#     neuron_indices, layer_positions, c=top_values, cmap='coolwarm',
#     edgecolor='k', s=80, vmin=-abs_max, vmax=abs_max
# )

# # Customize plot appearance
# ax.set_title(f"Top {top} Neurons per Layer - Neuron-wise Mean Difference ({model}-{size})",
#              fontsize=14, fontweight='bold', pad=15)
# ax.set_xlabel("Neuron Index", fontsize=12, fontweight='bold')
# ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
# ax.set_yticks(np.arange(1, mean_diff.shape[0] + 1))
# ax.tick_params(axis='both', which='major', labelsize=10)
# ax.grid(True, linestyle='--', alpha=0.5)

# # Add colorbar with label
# cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
# cbar.set_label("Neuron-wise Mean Difference (NMD)", fontsize=12, fontweight='bold')
# cbar.ax.tick_params(labelsize=10)

# plt.tight_layout()

# # Save the scatter plot
# save_path = os.path.join(save, f"top_{top}_neuron_positions_{task}_{size}.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# print(f"Scatter plot saved to:\n{save_path}")
# plt.show()

# -----------------------------------------------------------
# 5) Count frequency of top neurons indices across layers
# -----------------------------------------------------------
# Flatten the top_indices_matrix and count frequency for each neuron index
start = 1
end = 32

# Select only the indices within the specified layer range
selected_top_indices = top_indices_matrix[start:end].flatten()

# Count frequency for each unique neuron index
unique_indices, counts = np.unique(selected_top_indices, return_counts=True)

# Sort by frequency in descending order
sorted_order = np.argsort(-counts)
unique_indices_sorted = unique_indices[sorted_order]
counts_sorted = counts[sorted_order]

# Set frequency threshold to determine significant neurons
freq_threshold = 3  # 
filtered_indices = unique_indices_sorted[counts_sorted >= freq_threshold]
filtered_counts = counts_sorted[counts_sorted >= freq_threshold]

print(f"Neurons appearing at least {freq_threshold} times in layers {start}-{end}:")
for idx, neuron in enumerate(filtered_indices):
    print(f"Neuron index {neuron}: count = {filtered_counts[idx]}")

# Plot bar chart for neuron frequency in selected layer range
plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(filtered_indices)), filtered_counts,
        color='slateblue', edgecolor='black', linewidth=0.8)
plt.xticks(np.arange(len(filtered_indices)), filtered_indices,
           fontsize=8, rotation=45, ha='right')
plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
plt.ylabel("Frequency", fontsize=12, fontweight='bold')
plt.title(f"Neurons appearing ≥ {freq_threshold} times in Layers [{start}-{end}]", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()

# Save the frequency plot
freq_plot_path = os.path.join(save, f"significant_neurons_freq_{freq_threshold}_layers_{start}-{end}_{task}_{size}.png")
plt.savefig(freq_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Frequency plot saved to {freq_plot_path}")

# # -----------------------------------------------------------
# # 6) Scatter plot with significant neurons highlighted (using frequency threshold)
# # -----------------------------------------------------------
# # For example, we can set a frequency threshold
# start = 0
# end = 32
# freq_threshold = 5  # Change this to your desired frequency threshold

# selected_top_indices = top_indices_matrix[start:end].flatten()
# # Count frequency for each unique neuron index
# unique_indices, counts = np.unique(selected_top_indices, return_counts=True)

# #Select significant neurons based on the specified frequency
# significant_neurons = set(unique_indices[counts >= freq_threshold])
# print("Significant neurons (frequency >= {} in layers {}-{}):".format(freq_threshold, start, end), significant_neurons)

# # Redraw scatter plot for all layers
# fig, ax = plt.subplots(figsize=(12, 6))
# scatter = ax.scatter(
#     neuron_indices,    # Neuron indices for all layers
#     layer_positions,   # Layer positions for all layers (1-based)
#     c=top_values,
#     cmap='coolwarm',
#     edgecolor='k',
#     s=80,
#     vmin=-abs_max,
#     vmax=abs_max
# )

# # Highlight significant neurons only in the layer range [start+1, end]
# neuron_indices_arr = np.array(neuron_indices)
# layer_positions_arr = np.array(layer_positions)

# # (a) Create a mask to select neurons within the layer range [start+1, end]
# range_mask = (layer_positions_arr >= (start + 1)) & (layer_positions_arr <= end)

# # (b) Create a mask for the significant neurons (based on frequency threshold)
# sig_mask = np.isin(neuron_indices_arr, list(significant_neurons))

# # (c) Combine the two masks to highlight neurons that are both significant and within the layer range
# combined_mask = range_mask & sig_mask

# ax.scatter(
#     neuron_indices_arr[combined_mask],
#     layer_positions_arr[combined_mask],
#     facecolors='none',
#     edgecolors='lime',  # You can change this to "red", "cyan", etc. if desired
#     s=100,
#     linewidths=1.5,
# )

# # Set plot appearance
# ax.set_title(f"Significant Neurons per Layer with Partial Highlight (freq ≥ {freq_threshold}) [{start+1}-{end}] ({model}-{size})",
#              fontsize=14, fontweight='bold', pad=15)
# ax.set_xlabel("Neuron Index", fontsize=12, fontweight='bold')
# ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
# ax.set_yticks(np.arange(1, mean_diff.shape[0] + 1))
# ax.tick_params(axis='both', which='major', labelsize=10)
# ax.grid(True, linestyle='--', alpha=0.5)

# # Optional: Limit y-axis to the specified layer range (if desired)
# # ax.set_ylim(start + 0.5, end + 0.5)

# # Add colorbar
# cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
# cbar.set_label("Neuron-wise Mean Difference (NMD)", fontsize=12, fontweight='bold')
# cbar.ax.tick_params(labelsize=10)

# ax.legend()
# plt.tight_layout()

# # Save and display
# highlight_save_path = os.path.join(save, f"top_{top}_neuron_positions_highlight_significant_freq_{freq_threshold}_{task}_{size}.png")
# plt.savefig(highlight_save_path, dpi=300, bbox_inches='tight')
# print(f"Highlighted scatter plot saved to:\n{highlight_save_path}")
# plt.show()
