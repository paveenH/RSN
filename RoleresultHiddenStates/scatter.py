#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:41:36 2025

@author: paveenhuang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style: white background, "paper" context, and increased font scale
sns.set_theme(style="white", context="paper", font_scale=1.2)

### Load data ###
path = os.getcwd()
model = "phi"
size = "3.8B"
data_path = os.path.join(path, model)
save = os.path.join(path, "plot", model)
if not os.path.exists(save):
    os.makedirs(save)

task = "all_mean"
task_name = task.replace("_", " ")

data_char_diff = np.load(os.path.join(data_path, f"{task}_{size}.npy"))
data_none_char_diff = np.load(os.path.join(data_path, f"none_{task}_{size}.npy"))

num_samples = data_char_diff.shape[0]
num_time = data_char_diff.shape[1]
num_layers = data_char_diff.shape[2]
hidden_size = data_char_diff.shape[3]

print('char shape:', data_char_diff.shape)
print('none char shape:', data_none_char_diff.shape)
print(f"Data loaded successfully. Plots will be saved in: {save}")

# Compute difference and exclude layer 0 (embedding layer)
char_differences = data_char_diff - data_none_char_diff
char_differences = char_differences[:, :, 1:, :]
print('differences shape:', char_differences.shape)

### Plot top-N neurons scatter ###

# Step 1: Compute mean neuron-wise mean difference (NMD) for each neuron across samples
mean_diff = char_differences.mean(axis=0).squeeze()  # Shape: (layers, neurons)

# Determine top neurons: Here we use top = hidden_size//200
top = hidden_size // 200
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

print("Top indices matrix (per layer):")
print(top_indices_matrix)

# Step 3: Prepare scatter plot data
layer_positions = []
neuron_indices = []
top_values = []

for layer_idx, top_indices in top_neurons_per_layer:
    # Convert layer index to 1-based numbering for plotting
    layer_positions.extend([layer_idx + 1] * len(top_indices))
    neuron_indices.extend(top_indices)
    top_values.extend(mean_diff[layer_idx, top_indices])

# Calculate symmetric color limits for colorbar
abs_max = max(abs(np.min(top_values)), abs(np.max(top_values)))

# Step 4: Create scatter plot with publication-quality styling
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    neuron_indices, layer_positions, c=top_values, cmap='coolwarm',
    edgecolor='k', s=80, vmin=-abs_max, vmax=abs_max
)

# Customize plot appearance
ax.set_title(f"Top {top} Neurons per Layer - Neuron-wise Mean Difference ({model}-{size})",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Neuron Index", fontsize=12, fontweight='bold')
ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
ax.set_yticks(np.arange(1, mean_diff.shape[0] + 1))
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

# Add colorbar with label
cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label("Neuron-wise Mean Difference (NMD)", fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()

# Save the plot in both PDF and PNG formats
save_path = os.path.join(save, f"top_{top}_neuron_positions_{task}_{size}.pdf")
plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
png_path = save_path.replace('.pdf', '.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"Image saved to:\n{save_path}\n{png_path}")
plt.show()