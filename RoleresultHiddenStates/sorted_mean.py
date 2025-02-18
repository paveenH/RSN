#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:45:22 2025

@author: paveenhuang
"""


import numpy as np
import os
import matplotlib.pyplot as plt

path = os.getcwd()
model = "llama3"
size = "8B"
data_path = path + f"/{model}"
save = path + f"/plot/{model}"
if not os.path.exists(save):
    os.mkdir(save)

task = "all_mean"
task_name = task.replace("_", " ")

data_char_diff = np.load(f'{data_path}/{task}_{size}.npy')  
data_none_char_diff =  np.load(f'{data_path}/none_{task}_{size}.npy')  

num_samples = data_char_diff.shape[0]  
num_time = data_char_diff.shape[1]
num_layers = data_char_diff.shape[2]  
hidden_size = data_char_diff.shape[3]  

print('char shape:', data_char_diff.shape)
print('none char shape:', data_none_char_diff.shape)
print(f"Data loaded successfully. Plots will be saved in: {save}")

char_differences = data_char_diff - data_none_char_diff
char_differences = char_differences [:, :, 1:, :]
print('differences shape:', char_differences.shape)


# ---------------------------
# 1) Max and Mean Value
# ---------------------------
start = 0
end = num_layers

# # Max and Min samples and layers
max_char = data_char_diff[:, :, start:end, :].max(axis=(0, 3))  # Shape: (1, 33)
min_char = data_char_diff[:, :, start:end, :].min(axis=(0, 3))  # Shape: (1, 33)
max_none_char = data_none_char_diff[:, :, start:end, :].max(axis=(0, 3))  # Shape: (1, 33)
min_none_char = data_none_char_diff[:, :, start:end, :].min(axis=(0, 3))  # Shape: (1, 33)

layers = np.arange(start + 1, end + 1)  # Adjusted to reflect the selected range of layers

# Create a single figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot char max and min
ax.plot(layers, max_char.squeeze(), label='Char Max', color='blue', linestyle='-', marker='o')
ax.plot(layers, min_char.squeeze(), label='Char Min', color='blue', linestyle='--', marker='x')

# Plot none-char max and min
ax.plot(layers, max_none_char.squeeze(), label='None Char Max', color='orange', linestyle='-', marker='o')
ax.plot(layers, min_none_char.squeeze(), label='None Char Min', color='orange', linestyle='--', marker='x')

# Customize the plot
ax.set_title(f'Max and Min Activations at Different Layers ({task_name} expert {size})', fontsize=16)
ax.set_xlabel('Layers', fontsize=12)
ax.set_ylabel('Activation Values', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True)

plt.tight_layout()

# Save the plot
save_path = f"{save}/max_min_combined_activations_{task}_{size}.png"
plt.savefig(save_path, dpi=300)
print(f"Image has been saved to {save_path}")

# Show the plot
plt.show()

# ---------------------------
# 2) Mean Difference Without Order
# ---------------------------

# Step 1: Compute mean across samples for each layer
mean_diff = char_differences.mean(axis=0).squeeze()  # Shape: (layers, neurons)
# mean_diff = normalize_layer_v2(char_differences).mean(axis=0).squeeze() 

# Step 2: Sort neurons in each layer by mean difference
sorted_diff = np.sort(mean_diff, axis=1)  # Sort each row (layer) by neuron values

# Step 3: Calculate symmetric color limits
abs_max = max(abs(np.min(sorted_diff)), abs(np.max(sorted_diff)))

# Step 4: Plot the heatmap
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(sorted_diff, aspect='auto', cmap='seismic', origin='lower', vmin=-abs_max, vmax=abs_max)

# Add labels and title
ax.set_title(f"Sorted Neuron-Wise Mean Difference ({task_name} {size})", fontsize=14)
ax.set_xlabel("Neuron (sorted by mean difference)", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)

# Add colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Mean Difference", fontsize=12)

# Save the plot
save_path = f"{save}/sorted_neuron_mean_difference_{task}_{size}.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {save_path}")
plt.show()