# Pranav Minasandra
# 26 Mar 2026
# pminasandra.github.io

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

def percentile_bin_to_center(percentile_bins, thresholds=config.PERCENTILE_THRESHOLDS):
    """
    Map percentile-bin upper bounds to bin centers.

    Example:
        thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]
        bin 0.4 → center = (0.2 + 0.4) / 2 = 0.3

    Args:
        percentile_bins (array-like or pd.Series): values like 0.2, 0.4, ...
        thresholds (array-like): same thresholds used to define bins

    Returns:
        np.ndarray or pd.Series: bin centers
    """
    thresholds = np.asarray(thresholds, dtype=float)
    lower_bounds = np.concatenate(([0.0], thresholds[:-1]))

    # build mapping upper → center
    centers = (lower_bounds + thresholds) / 2
    mapping = dict(zip(thresholds, centers))

    if isinstance(percentile_bins, pd.Series):
        return percentile_bins.map(mapping)

    return np.array([mapping.get(x, np.nan) for x in percentile_bins])


# 19.08.2026


def _draw_line_with_ci(ax, plotdf, label, color=None):
    """
    Helper - normal approx: p_estimate +/- 1.96 * p_error
    """
    Z95 = 1.96  # normal-approx 95% CI multiplier on the bootstrap SE (p_error)
    plotdf = plotdf.sort_values("percentile_bin")

    sns.lineplot(
        data=plotdf, x="percentile_bin", y="p_estimate",
        marker="o", linewidth=0.7, ax=ax, label=label, color=color, errorbar=None,
    )
    ax.errorbar(
        plotdf["percentile_bin"], plotdf["p_estimate"], yerr=plotdf["p_error"],
        fmt="none", color=color, capsize=2, linewidth=0.6,
    )
    ax.fill_between(
        plotdf["percentile_bin"],
        plotdf["p_estimate"] - Z95 * plotdf["p_error"],
        plotdf["p_estimate"] + Z95 * plotdf["p_error"],
        color=color, alpha=0.15, linewidth=0,
    )


def plot_with_ci(results: pd.DataFrame):
    """
    Re-plots output of analyses.analyse_sleep_wake_asymmetry_by, adding a
    shaded 95% CI band around each line

    Args:
        results (pd.DataFrame): output of analyses.analyse_sleep_wake_asymmetry_by
    Returns:
        plt.Figure
        plt.Axes or np.ndarray of Axes
    """
    sns.set_theme(style="whitegrid")
    single_panel = results["label"].nunique() == 1

    if single_panel:
        fig, ax = plt.subplots(figsize=(6, 4))
        for eventtype in ["sleep", "wake"]:
            _draw_line_with_ci(ax, results[results["eventtype"] == eventtype], eventtype)
        ax.set_title("sleep and wake")
        ax.set_xlabel("percentile_bin")
        ax.set_ylabel("p_estimate")
        ax.legend(fontsize=8, title_fontsize=9, frameon=False)
        fig.tight_layout()
        return fig, ax

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    event_axes = {"sleep": axes[0], "wake": axes[1]}
    labels = list(pd.unique(results["label"]))
    palette = sns.color_palette(n_colors=max(len(labels), 1))
    color_map = dict(zip(labels, palette))

    for eventtype in ["sleep", "wake"]:
        ax = event_axes[eventtype]
        for label, subdf in results[results["eventtype"] == eventtype].groupby("label"):
            _draw_line_with_ci(ax, subdf, label, color_map[label])
        ax.set_title(eventtype)
        ax.set_xlabel("percentile_bin")
        ax.set_ylabel("p_estimate")
        ax.legend(fontsize=8, title_fontsize=9, frameon=False)

    fig.tight_layout()
    return fig, axes

'''   
    Yet to decide reasonable panel layout!!
    
    depends on results['label']:  "all" label gives one panel with sleep/wake
    overlaid; any other labels (so split by site/age-sex/group) two-panel layout, one
    line per label. 
    
'''
 