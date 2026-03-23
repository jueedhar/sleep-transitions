# Pranav Minasandra
# 23 Mar 2026
# pminasandra.github.io

"""
Functions to compute transition probabilities based on num already awake
"""

import math
from typing import List, Dict

import numpy as np
import pandas as pd

def accumulate_percentile_window_durations(
    df: pd.DataFrame,
    eventtype: str,
    perc_thresholds: List[float],
    min_tags: int,
) -> Dict[float, List[float]]:
    """
    For each date × clutch_id subset, find the ordered transition timestamps for the
    requested event type ("wake" or "sleep"), then compute the elapsed duration
    spanned by each percentile-to-percentile window.

    Percentiles are based on event order through time.

    Example:
        transition times = [21600, 21620, 21670, 21710, 21800]
        perc_thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]

        n = 5, so cutoff indices are:
            0.2 -> ceil(0.2 * 5) - 1 = 0
            0.4 -> ceil(0.4 * 5) - 1 = 1
            0.6 -> ceil(0.6 * 5) - 1 = 2
            0.8 -> ceil(0.8 * 5) - 1 = 3
            1.0 -> ceil(1.0 * 5) - 1 = 4

        corresponding timestamps:
            [21600, 21620, 21670, 21710, 21800]

        window durations added to the dict:
            0.2 -> 21600 - 21600 = 0
            0.4 -> 21620 - 21600 = 20
            0.6 -> 21670 - 21620 = 50
            0.8 -> 21710 - 21670 = 40
            1.0 -> 21800 - 21710 = 90

    Args:
        df (pd.DataFrame): master data frame
        eventtype (str): "wake" or "sleep"
        perc_thresholds (List[float]): increasing percentile cutoffs in (0, 1],
            e.g. [0.2, 0.4, 0.6, 0.8, 1.0]
        min_tags (int): minimum number of observed transitions in a date × clutch_id
            subset to include that subset

    Returns:
        Dict[float, List[float]]: maps each percentile-window upper bound to a list
        of elapsed durations for that window across all date × clutch_id subsets
    """
    if eventtype not in {"wake", "sleep"}:
        raise ValueError("eventtype must be either 'wake' or 'sleep'")

    if not perc_thresholds:
        raise ValueError("perc_thresholds must not be empty")

    if any(p <= 0 or p > 1 for p in perc_thresholds):
        raise ValueError("All percentile thresholds must satisfy 0 < p <= 1")

    if any(perc_thresholds[i] <= perc_thresholds[i - 1]
           for i in range(1, len(perc_thresholds))):
        raise ValueError("perc_thresholds must be strictly increasing")

    if perc_thresholds[-1] != 1.0:
        raise ValueError("The last percentile threshold must be 1.0")

    timecol = f"t_{eventtype}"
    if timecol not in df.columns:
        raise ValueError(f"Column '{timecol}' not found in dataframe")

    out: Dict[float, List[float]] = {p: [] for p in perc_thresholds}

    for (_, _), subdf in df.groupby(["date", "clutch_id"]):
        times = np.sort(subdf[timecol].dropna().to_numpy())

        n = len(times)
        if n < min_tags:
            continue

        prev_time = times[0]

        for p in perc_thresholds:
            idx = math.ceil(p * n) - 1
            idx = min(max(idx, 0), n - 1)

            current_time = times[idx]
            out[p].append(float(current_time - prev_time))
            prev_time = current_time

    return out

def unbiased_exp_param_estimate(data: List[float]) -> float:
    """
    From a given set of durations, returns the de-biased MLE of the geometric
    distribution that best fits it. Estimator p_hat is 1/x.avg. Bias is p_hat(1-p_hat)/n.

    Args:
        data (List[float]): all durations to consider

    Returns:
        float: unbiased estimate of exponential rate parameter (lambda)
    """
    if len(data) < 2:
        raise ValueError("Need at least two data points for unbiased estimate")

    arr = np.asarray(data, dtype=float)

    if np.any(arr <= 0):
        raise ValueError("All durations must be positive for exponential fitting")

    mean_val = arr.mean()
    if mean_val == 0:
        raise ValueError("Mean of data is zero, cannot compute estimate")

    n = len(arr)
    p_hat = 1/mean_val
    bias_hat = p_hat*(1-p_hat)/n
    debiased_p_hat = p_hat - bias_hat

    return debiased_p_hat #removing biasing


def unbiased_exp_param_sd(data: List[float], n_boot: int = 100) -> float:
    """
    Estimates the standard deviation (standard error) of the unbiased geometric
    rate parameter estimator using bootstrap resampling.

    For each bootstrap replicate, the data are resampled with replacement, the
    unbiased geometric rate parameter is computed, and the standard deviation
    of these estimates is returned.

    Args:
        data (List[float]): observed durations (must be positive)
        n_boot (int): number of bootstrap replicates (default: 1000)

    Returns:
        float: bootstrap estimate of the standard deviation of the estimator
    """
    arr = np.asarray(data, dtype=float)
    if arr.size < 2:
        raise ValueError("Need at least two data points for bootstrap")

    if np.any(arr <= 0):
        raise ValueError("All durations must be positive")

    estimates = np.empty(n_boot, dtype=float)
    n = arr.size

    for i in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        estimates[i] = unbiased_exp_param_estimate(sample)

    return estimates.std(ddof=1)


def estimate_exp_by_percentile_df(
    percentile_durations: Dict[float, List[float]],
    n_boot: int = 100
) -> pd.DataFrame:
    """
    For each percentile window, compute the unbiased exponential rate parameter
    estimate and its bootstrap standard deviation, and return results as a DataFrame.

    Args:
        percentile_durations (Dict[float, List[float]]): mapping from percentile
            window upper bound to the list of durations assigned to that window,
            typically the output of accumulate_to_percentiles()
        n_boot (int): number of bootstrap replicates for SD estimation

    Returns:
        pd.DataFrame with columns:
            - percentile: percentile window upper bound
            - estimate: unbiased exponential rate parameter estimate
            - sd: bootstrap standard deviation of the estimate
            - n: number of durations in that percentile bin
    """
    rows = []

    for perc, durations in percentile_durations.items():
        n = len(durations)

        if n < 2:
            estimate = np.nan
            sd = np.nan
        else:
            estimate = unbiased_exp_param_estimate(durations)
            sd = unbiased_exp_param_sd(durations, n_boot=n_boot)

        rows.append({
            "percentile": perc,
            "estimate": estimate,
            "sd": sd,
            "n": n,
        })

    df = pd.DataFrame(rows)

    # Sort by percentile for readability
    df = df.sort_values("percentile").reset_index(drop=True)

    return df
