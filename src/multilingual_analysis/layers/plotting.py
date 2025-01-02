"""Plotting fuctions for layer analysis."""
from pathlib import PosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_line_plot(language_stats: list[dict], save_path: PosixPath) -> None:
    """Create a detailed lineplot for en and de tokens using seaborn."""
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

    # Prepare data for seaborn
    data_for_plot = []
    for key, values in means.items():
        data_for_plot.append({
            "key": key,
            "language": "en",
            "mean_value": values["mean_en"]
        })
        data_for_plot.append({
            "key": key,
            "language": "de",
            "mean_value": values["mean_de"]
        })

    # Convert to DataFrame
    df = pd.DataFrame(data_for_plot)

    # Plot the data using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="key", y="mean_value", hue="language", markers=True)

    # Customize the plot
    plt.xlabel("Keys")
    plt.ylabel("Mean Values")
    plt.title("Mean Values of en and de by Key")
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path)
    plt.close()


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
