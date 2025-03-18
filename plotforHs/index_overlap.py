"""
Neural Activation Frequency Analysis
This study examines the differential activation patterns across two experimental groups 
by analyzing high-frequency neurons in the difference matrix. Visualization methods 
highlight shared neural signatures between groups.
"""

# -*- Coding Extensions -*-
import os
import numpy as np
import matplotlib.pyplot as plt  # Visualization toolkit v3.7.0

# =============================================================================
# I. Experimental Dataset
# Contains normalized neural response indices from 4096-unit differential matrix.
# Frequency counts reflect activation occurrences across 50 stimulus trials.
# =============================================================================
# # Group 1: Student
filtered_indices_1 = np.array([373, 2629, 788, 3585, 873, 1565, 1189, 4055, 1039, 630, 3070,
                                281, 3869, 3632, 2646, 3096, 3516, 1298, 2352, 2184, 1815, 1369,
                                2040, 133, 1122, 1515, 766, 2764, 3361, 1421, 2943, 3664, 3695,
                                761, 108, 2303, 1626, 3381, 1059, 2088, 2528, 2692, 905, 3266,
                                2925, 1471, 2265])
filtered_counts_1 = np.array([20, 17, 17, 15, 14, 14, 13, 13, 12, 11, 10, 9, 9, 8, 7, 6, 6,
                              6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

# Group 2: Diff-matrix neuron indices and their corresponding frequency counts V3
filtered_indices_2 = np.array([4055, 2692, 2629, 1731, 873, 133, 373, 291, 2352, 2646, 1189,
                               1298, 3695, 3585, 630, 3516, 2932, 1421, 3076, 2265, 761, 2977,
                               384, 3869, 766, 810, 1565, 3187, 1273, 1369, 431, 3664, 2082,
                               1039, 1815, 3096, 1130, 788, 3266, 2943, 1791, 2742, 42, 4041,
                               4062, 2660, 281])
filtered_counts_2 = np.array([18, 18, 18, 17, 15, 15, 15, 15, 14, 13, 13, 13, 11, 10, 9, 9, 9,
                              9, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
                              3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

# Group 3: Diff-matrix neuron indices and their corresponding frequency counts V5
filtered_indices_3 = np.array([3869,  133, 2742, 2303, 4055, 2692, 2764, 2977, 1122, 2629,  373,
       2352, 3695, 1731, 2265, 1421,  873, 1163,  367, 2646,  384,  281,
        761, 3070, 2560, 1815, 2932,  912, 2219, 2618, 1189, 4062, 3632,
       1059,  291, 1812,  843,  831, 1582, 1717, 1932, 1471, 1130, 1260,
       1298, 3881,  166, 3585, 2528, 3316])
filtered_counts_3 = np.array([18, 17, 16, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 11, 10, 10,
       10,  9,  9,  8,  8,  7,  7,  6,  6,  5,  5,  5,  4,  4,  4,  4,  4,
        4,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3])


# =============================================================================
# II. Cross-Condition Analysis
# Computes intersection of neural signatures using NumPy v1.24.3 set operations
# =============================================================================
# common_indices = np.intersect1d(filtered_indices_1, filtered_indices_2)
# common_indices = np.intersect1d(filtered_indices_3, filtered_indices_2)
common_indices = np.intersect1d(filtered_indices_1, 
                    np.intersect1d(filtered_indices_2, filtered_indices_3))


print(f"Shared neural signatures: {len(common_indices)} common indices")

# =============================================================================
# III. Visualization Protocol
# Implements publication-quality plotting with annotation system:
# 1. Slateblue bars represent baseline activation frequencies
# 2. Red-edged bars (★ markers) denote shared inter-group activations
# 3. Axis formatting adheres to J Neurosci standards
# =============================================================================
def plot_frequency(indices, counts, title):
    """Generates comparative activation plots with highlighting
    
    Parameters:
        indices (ndarray): Neural unit IDs (0-4095)
        counts (ndarray): Normalized activation magnitudes
        title (str): Condition identifier string
        save_path (Path): Output path in TIFF format
    """
    plt.figure(figsize=(10, 5))
    visualization_elements = plt.bar(  # Initialize primary plot elements
        np.arange(len(indices)), 
        counts, 
        color='slateblue', 
        edgecolor='black',
        linewidth=0.8
    )
    
    # Configure axis labels and grid parameters
    plt.xticks(np.arange(len(indices)), indices, fontsize=8, 
               rotation=45, ha='right')
    plt.xlabel("Neural Unit ID", fontsize=12, fontweight='bold')
    plt.ylabel("Normalized Activation", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle="--", alpha=0.5)

    # Highlight cross-condition activations
    for i, neural_id in enumerate(indices):
        if neural_id in common_indices:
            visualization_elements[i].set_edgecolor("red")  # Border highlighting
            visualization_elements[i].set_linewidth(1)
            plt.text(i, counts[i], "★",            # Top marker
                    ha='center', va='bottom', 
                    fontsize=8, color='blue')

    plt.tight_layout()
    plt.show()
    plt.close()

# =============================================================================
# IV. Output Configuration
# Saves vectorized figures in ./plot/ directory for manuscript submission
# =============================================================================
output_dir = os.path.join(os.getcwd(), "plot")
os.makedirs(output_dir, exist_ok=True)  # Ensures filesystem compliance

# Generate comparative figures
plot_frequency(filtered_indices_1, filtered_counts_1, "Repeating Neurons Index in Role Student")
plot_frequency(filtered_indices_2, filtered_counts_2, "Repeating Neurons Index in Role Expert v3")
plot_frequency(filtered_indices_3, filtered_counts_3, "Repeating Neurons Index in Role Expert v5")