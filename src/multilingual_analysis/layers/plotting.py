"""Plotting fuctions for layer analysis."""
from pathlib import PosixPath

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_line_plot(language_stats:list[dict],save_path:PosixPath) -> None:
    """Create a detailed lineplot for en and de tokens."""
    # Initialize dictionaries to store cumulative sums and counts
    sums = {}
    counts = {}

    # Iterate through each dictionary in the list
    for data in language_stats:
        for key, values in data.items():
            if key not in sums:
                sums[key] = {"en": 0, "de": 0}
                counts[key] = 0
            sums[key]["en"] += values["en"]
            sums[key]["de"] += values["de"]
            counts[key] += 1

    # Calculate the mean for each key
    means = {
        key: {
            "mean_en": sums[key]["en"] / counts[key],
            "mean_de": sums[key]["de"] / counts[key],
        }
        for key in sums
    }

    # Extract x-axis values and mean values for 'en' and 'de'
    x = list(means.keys())
    mean_en_values = [means[k]["mean_en"] for k in x]
    mean_de_values = [means[k]["mean_de"] for k in x]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_en_values, label="Mean en", marker="o")
    plt.plot(x, mean_de_values, label="Mean de", marker="o")

    # Explicitly set x-axis ticks
    plt.xticks(ticks=x)  # Ensures all x values are displayed

    # Customize the plot
    plt.xlabel("Keys")
    plt.ylabel("Mean Values")
    plt.title("Mean Values of en and de by Key")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)


def plot_lang_distribution(lang_distribution, candidate_langs:list[str], candidate_layers:list[int],save_path:PosixPath) -> None:
    """Create a distribution plot."""
    lang_distribution_matrix= [[lang_distribution[layer][lang] for lang in candidate_langs]for layer in candidate_layers]
    lang_distribution_matrix = np.array(lang_distribution_matrix).T
    fig, ax = plt.subplots(figsize=(11, 3))
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    sns.heatmap(lang_distribution_matrix, ax=ax, xticklabels=candidate_layers, yticklabels=candidate_langs, cmap=cmap)
    plt.title("Layerwise Language Distribution")
    plt.xlabel("Layer")
    plt.ylabel("Language")
    plt.savefig(save_path)
