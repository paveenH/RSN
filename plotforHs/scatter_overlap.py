import os
import numpy as np
import matplotlib.pyplot as plt

### (1) Example: Define paths and parameters (modify according to actual setup) ###
path = os.getcwd()
model_v3 = "llama3_v3"
model_v5 = "llama3_v5"
size = "8B"
task = "all_mean"

data_path_v3 = os.path.join(path, model_v3)
data_path_v5 = os.path.join(path, model_v5)

save = os.path.join(path, "plot", "compare_v3_v5")
if not os.path.exists(save):
    os.makedirs(save)

### (2) Load v3 data ###
data_char_diff_v3 = np.load(os.path.join(data_path_v3, f"{task}_{size}.npy"))
data_none_char_diff_v3 = np.load(os.path.join(data_path_v3, f"none_{task}_{size}.npy"))

char_differences_v3 = data_char_diff_v3 - data_none_char_diff_v3
# Remove the 0th layer (embedding layer)
char_differences_v3 = char_differences_v3[:, :, 1:, :]

# Assuming the shape is (num_samples, num_time, num_layers, hidden_size)
print("v3 char_differences shape:", char_differences_v3.shape)

# Compress the first two dimensions (as done in your original code .squeeze(0).squeeze(0))
mean_diff_v3 = char_differences_v3.squeeze(0).squeeze(0)  # (num_layers, hidden_size)
print("v3 mean_diff shape:", mean_diff_v3.shape)

### (3) Load v5 data ###
data_char_diff_v5 = np.load(os.path.join(data_path_v5, f"{task}_{size}.npy"))
data_none_char_diff_v5 = np.load(os.path.join(data_path_v5, f"none_{task}_{size}.npy"))

char_differences_v5 = data_char_diff_v5 - data_none_char_diff_v5
char_differences_v5 = char_differences_v5[:, :, 1:, :]

print("v5 char_differences shape:", char_differences_v5.shape)

mean_diff_v5 = char_differences_v5.squeeze(0).squeeze(0)  # (num_layers, hidden_size)
print("v5 mean_diff shape:", mean_diff_v5.shape)


# -------------------------------------------------------------
# (4) Select top-N neurons for both v3 and v5
# -------------------------------------------------------------
top = 20

num_layers_v3, hidden_size_v3 = mean_diff_v3.shape
num_layers_v5, hidden_size_v5 = mean_diff_v5.shape

top_indices_matrix_v3 = np.zeros((num_layers_v3, top), dtype=int)
top_values_matrix_v3 = np.zeros((num_layers_v3, top))

top_indices_matrix_v5 = np.zeros((num_layers_v5, top), dtype=int)
top_values_matrix_v5 = np.zeros((num_layers_v5, top))

# Top-N for v3
for layer_idx in range(num_layers_v3):
    layer_values = mean_diff_v3[layer_idx]
    # Sort by absolute value, then select the top-N
    top_indices = np.argsort(np.abs(layer_values))[-top:]
    top_indices_matrix_v3[layer_idx] = top_indices
    top_values_matrix_v3[layer_idx] = layer_values[top_indices]

# Top-N for v5
for layer_idx in range(num_layers_v5):
    layer_values = mean_diff_v5[layer_idx]
    top_indices = np.argsort(np.abs(layer_values))[-top:]
    top_indices_matrix_v5[layer_idx] = top_indices
    top_values_matrix_v5[layer_idx] = layer_values[top_indices]

print(f"\nTop-{top} neurons for v3 -> shape: {top_indices_matrix_v3.shape}")
print(f"Top-{top} neurons for v5 -> shape: {top_indices_matrix_v5.shape}")


# -------------------------------------------------------------
# (5) Plot three types of scatter plots on the same graph:
#     - v3 only (blue)
#     - v5 only (red)
#     - v3 & v5 overlap (purple)
# -------------------------------------------------------------
overlap_indices = []
overlap_layers = []

v3_only_indices = []
v3_only_layers = []

v5_only_indices = []
v5_only_layers = []

# Assume the number of layers is the same for both v3 and v5
num_layers = min(num_layers_v3, num_layers_v5)

for layer_idx in range(num_layers):
    set_v3 = set(top_indices_matrix_v3[layer_idx])
    set_v5 = set(top_indices_matrix_v5[layer_idx])
    common = set_v3.intersection(set_v5)  # Overlap
    v3_only = set_v3 - common            # v3 only
    v5_only = set_v5 - common            # v5 only

    # Record overlapping neurons
    for neuron in common:
        overlap_indices.append(neuron)
        overlap_layers.append(layer_idx + 1)  # 1-based

    # Record v3-only neurons
    for neuron in v3_only:
        v3_only_indices.append(neuron)
        v3_only_layers.append(layer_idx + 1)

    # Record v5-only neurons
    for neuron in v5_only:
        v5_only_indices.append(neuron)
        v5_only_layers.append(layer_idx + 1)

print("\nNumber of overlap neurons:", len(overlap_indices))
print("Number of v3-only neurons:", len(v3_only_indices))
print("Number of v5-only neurons:", len(v5_only_indices))


# -------------------------------------------------------------
# (6) Plot the scatter plot for the three types + legend (optimize colors and legend position)
# -------------------------------------------------------------
plt.figure(figsize=(12, 6), dpi=300)

# v3-only: Use light blue (alpha set to 0.7)
plt.scatter(v3_only_indices, v3_only_layers,
            color='lightblue', edgecolor='k', s=80, label='v3-only', alpha=0.7)

# v5-only: Use light red (alpha set to 0.7)
plt.scatter(v5_only_indices, v5_only_layers,
            color='lightcoral', edgecolor='k', s=80, label='v5-only', alpha=0.7)

# overlap: Use darker purple
plt.scatter(overlap_indices, overlap_layers,
            color='purple', edgecolor='k', s=80, label='Overlap')

plt.xlabel(f"Neuron Index (top-{top})", fontsize=12, fontweight='bold')
plt.ylabel("Layer", fontsize=12, fontweight='bold')
plt.title(f"Compare Top-{top} Neurons: v3 vs v5\n(Overlap & Non-overlap)", 
          fontsize=14, fontweight='bold', pad=12)
plt.yticks(np.arange(1, num_layers + 1))

plt.grid(True, linestyle="--", alpha=0.5)
# Adjust legend position, place it in the upper right corner
plt.legend(fontsize=10, loc='upper right', frameon=True)
plt.tight_layout()

output_file = os.path.join(save, f"compare_top_{top}_neurons_v3_vs_v5_overlap.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print("Scatter plot saved to:", output_file)

# ---------------------------
# 7) Count the frequency of repeated neuron index occurrences for each layer
# ---------------------------
# For v3, count how many layers each neuron index appears in
unique_v3, counts_v3 = np.unique(top_indices_matrix_v3, return_counts=True)
# For v5, count how many layers each neuron index appears in
unique_v5, counts_v5 = np.unique(top_indices_matrix_v5, return_counts=True)

# For overlapping top-neurons (neuron indices appearing in both v3 and v5 for each layer), count their frequency
overlap_indices_all = []
for layer_idx in range(num_layers):
    top_v3 = set(top_indices_matrix_v3[layer_idx])
    top_v5 = set(top_indices_matrix_v5[layer_idx])
    overlap = top_v3.intersection(top_v5)
    overlap_indices_all.extend(list(overlap))

unique_overlap, counts_overlap = np.unique(overlap_indices_all, return_counts=True)
print("\nOverlap Top-Neurons frequency statistics:")

# Keep only neuron indices that appear more than the threshold (threshold=3)
threshold = 3

mask_v3 = counts_v3 > threshold
unique_v3_filtered = unique_v3[mask_v3]
counts_v3_filtered = counts_v3[mask_v3]

mask_v5 = counts_v5 > threshold
unique_v5_filtered = unique_v5[mask_v5]
counts_v5_filtered = counts_v5[mask_v5]

mask_overlap = counts_overlap > threshold
unique_overlap_filtered = unique_overlap[mask_overlap]
counts_overlap_filtered = counts_overlap[mask_overlap]

# ---------------------------
# (Optional) Plot histograms to show frequency (only for neurons with frequency > 3)
# ---------------------------
plt.figure(figsize=(12, 4), dpi=300)

plt.subplot(1, 3, 1)
plt.bar(unique_v3_filtered, counts_v3_filtered, color='lightblue', edgecolor='k')
plt.xlabel("Neuron Index")
plt.ylabel("Appearance in Layers")
plt.title("v3 Top-Neurons Frequency (> 3)")

plt.subplot(1, 3, 2)
plt.bar(unique_v5_filtered, counts_v5_filtered, color='lightcoral', edgecolor='k')
plt.xlabel("Neuron Index")
plt.title("v5 Top-Neurons Frequency (> 3)")

plt.subplot(1, 3, 3)
plt.bar(unique_overlap_filtered, counts_overlap_filtered, color='purple', edgecolor='k')
plt.xlabel("Neuron Index")
plt.title("Overlap Top-Neurons Frequency (> 3)")

plt.tight_layout()
output_freq = os.path.join(save, "top_neurons_frequency_threshold3.png")
plt.savefig(output_freq, dpi=300, bbox_inches="tight")
plt.show()
print(f"Frequency plot saved to: {output_freq}")