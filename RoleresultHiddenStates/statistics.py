#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:07:11 2025

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