#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:42:36 2025

@author: paveenhuang
"""

### Calculate Pearson correlation across layers ###

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set theme for publication quality (white background, paper context, and increased font scale)
sns.set_theme(style="white", context="paper", font_scale=1.2)

### Load data ###
path = os.getcwd()
model = "phi_v4"
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

# Compute difference and exclude the embedding layer (layer 0)
char_differences = data_char_diff - data_none_char_diff
char_differences = char_differences[:, :, 1:, :]
print('differences shape:', char_differences.shape)

# Average over samples and time --> shape: (num_layers-1, neurons)
reshaped_data = char_differences.mean(axis=0).squeeze()

# Compute Pearson correlation matrix across layers
pearson_corr_matrix = np.corrcoef(reshaped_data)
print("Pearson correlation matrix between layers (excluding layer 0):")

# Plot heatmap with publication-quality style
plt.figure(figsize=(10, 8))
ax = sns.heatmap(pearson_corr_matrix, annot=False, cmap="coolwarm", 
                 xticklabels=range(1, num_layers), 
                 yticklabels=range(1, num_layers),
                 cbar_kws={'label': 'Pearson r'})
ax.set_title(f"Layer-wise Pearson Correlation ({model}-{size})", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Layer", fontsize=14, fontweight='bold')
ax.set_ylabel("Layer", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save, f"pearson_correlation_{model}_{size}.png"), dpi=300, bbox_inches='tight')
plt.show()