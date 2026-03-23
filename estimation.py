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

def accumulate_to_percentiles(
    durations_list: List[np.ndarray],
    perc_thresholds: List[float],
    min_tags: int
) -> Dict[float, List[float]]:
    """
    Takes list of inter-event durations on each day for each clutch, and then adds each
    duration into one list for each percentile-to-percentile window. Percentile matching
    is uniquely done for each day for each clutch.

    Args:
        durations_list (List[np.ndarray]): output from
            durations.get_transition_duration_table()
        perc_thresholds (List[float]): upper bounds of percentile windows.
            E.g., [0.1, 0.5, 0.7, 0.9, 1.0]
        min_tags (int): minimum number of points in an array in durations_list to
            consider it for analysis

    Returns:
        Dict[float, List[float]]: maps each percentile-window upper bound to a list
        of all durations accumulated into that window across clutch-days.
        For example, key 0.5 contains durations in the 0.1–0.5 window if the
        previous threshold was 0.1.
    """
    if not perc_thresholds:
        raise ValueError("perc_thresholds must not be empty")

    perc_thresholds = list(perc_thresholds)

    if any(p <= 0 or p > 1 for p in perc_thresholds):
        raise ValueError("All percentile thresholds must satisfy 0 < p <= 1")

    if any(perc_thresholds[i] <= perc_thresholds[i - 1]
           for i in range(1, len(perc_thresholds))):
        raise ValueError("perc_thresholds must be strictly increasing")

    if perc_thresholds[-1] != 1.0:
        raise ValueError("The last percentile threshold must be 1.0")

    out: Dict[float, List[float]] = {p: [] for p in perc_thresholds}

    for durations in durations_list:
        durations = np.asarray(durations)

        if len(durations) < min_tags:
            continue

        durations_sorted = np.sort(durations)
        n = len(durations_sorted)

        prev_idx = 0
        for p in perc_thresholds:
            end_idx = math.ceil(p * n)

            # Guard against any floating-point oddities
            end_idx = min(max(end_idx, prev_idx), n)

            window_vals = durations_sorted[prev_idx:end_idx]
            out[p].extend(window_vals.tolist())

            prev_idx = end_idx

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

    return debiased_p_hat


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
