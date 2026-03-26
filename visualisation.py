# Pranav Minasandra
# 26 Mar 2026
# pminasandra.github.io

import numpy as np
import pandas as pd

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
