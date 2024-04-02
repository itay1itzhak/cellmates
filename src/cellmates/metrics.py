from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_calibration(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_cells_per_bin: int = 1000,
    ax=None,
    max_p=1.0,
    plot_min_max_lines: bool = False,
):
    n_bins = int(len(predicted_probs) / n_cells_per_bin)
    bin_true_p, bin_pred_p = calibration_curve(
        true_labels,
        predicted_probs,
        n_bins=n_bins,
        strategy="quantile",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    sns.lineplot(
        x=np.linspace(-0.1, max_p, 10),
        y=np.linspace(-0.1, max_p, 10),
        color="black",
        linestyle="--",
        ax=ax,
    )
    sns.scatterplot(
        y=bin_true_p, x=bin_pred_p, color="#53AC69", edgecolor="#252526", ax=ax
    )

    if plot_min_max_lines:
        ax.axhline(y=bin_true_p.min(), color="red", linestyle="--")
        ax.axhline(y=bin_true_p.max(), color="red", linestyle="--")

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("true probability")

    # add n_bins
    for i in range(n_bins):
        ax.axvline(i / n_bins, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlim(-0.1, max_p)
    ax.set_ylim(-0.1, max_p)
    sns.despine(ax=ax)

    return fig
