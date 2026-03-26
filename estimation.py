# Pranav Minasandra
# 23 Mar 2026
# pminasandra.github.io

"""
Functions to compute transition probabilities based on num already awake
"""

import math
from typing import List, Dict, Sequence

import numpy as np
import pandas as pd


def assign_percentile_bin(
    values: Sequence[float],
    perc_thresholds: Sequence[float],
    return_upper: bool = True
):
    """
    Assign each value (proportion in [0, 1]) to a percentile bin defined by
    increasing thresholds.

    Bins are interpreted as:
        (0, p1], (p1, p2], ..., (p_{k-1}, p_k]

    Args:
        values (array-like): np.ndarray or pd.Series of proportions in [0, 1]
        perc_thresholds (array-like): sorted thresholds in (0, 1], e.g.
            [0.2, 0.4, 0.6, 0.8, 1.0]
        return_upper (bool): if True, return the upper bound of the bin (float).
            if False, return bin indices (int, 0-based)

    Returns:
        np.ndarray or pd.Series: same type as input, containing bin labels
    """
    thresholds = np.asarray(perc_thresholds)

    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("perc_thresholds must be a 1D non-empty array")

    if np.any(thresholds <= 0) or np.any(thresholds > 1):
        raise ValueError("thresholds must satisfy 0 < p <= 1")

    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("thresholds must be strictly increasing")

    # Convert input to array for computation
    is_series = isinstance(values, pd.Series)
    arr = values.to_numpy() if is_series else np.asarray(values)

    # Find insertion indices: gives bin index
    # side='left' ensures values == threshold go to that bin
    idx = np.searchsorted(thresholds, arr, side="left")

    # Values above last threshold (shouldn't happen if max=1) → last bin
    idx = np.clip(idx, 0, len(thresholds) - 1)

    if return_upper:
        result = thresholds[idx]
    else:
        result = idx

    if is_series:
        return pd.Series(result, index=values.index)
    return result

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

def get_estimates_of_p_each_n(df: pd.DataFrame, n_boot:int = 100) -> Dict[int, pd.DataFrame]:
    """
    For each value of n_left, use unbiased_exp_param_estimate on the interval
    durations within each percentile bin.

    Assumes the input dataframe contains at least:
        - 'interval_dur'
        - 'n_left'
        - 'percentile_bin'

    Args:
        df (pd.DataFrame): output from durations.get_transition_duration_table(...),
            after adding a 'percentile_bin' column
        n_boot (int): passed to unbiased_exp_param_sd

    Returns:
        Dict[int, pd.DataFrame]: maps each n_left to a dataframe with columns:
            - 'percentile_bin'
            - 'p_estimate'
            - 'p_error'
            - 'n_data_points'
    """
    required_cols = {"interval_dur", "n_left", "percentile_bin"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataframe is missing required columns: {sorted(missing)}"
        )

    out: Dict[int, pd.DataFrame] = {}

    df_valid = df.dropna(subset=["interval_dur", "n_left", "percentile_bin"]).copy()

    for n_left, subdf in df_valid.groupby("n_left"):
        rows = []

        for percentile_bin, subsubdf in subdf.groupby("percentile_bin"):
            data = subsubdf["interval_dur"].to_numpy(dtype=float)
            n_data_points = len(data)

            if n_data_points < 2:
                p_estimate = np.nan
            else:
                p_estimate = unbiased_exp_param_estimate(data.tolist())
                p_error = unbiased_exp_param_sd(data.tolist(), n_boot=n_boot)

            rows.append(
                {
                    "percentile_bin": percentile_bin,
                    "p_estimate": p_estimate,
                    "p_error": p_error,
                    "n_data_points": n_data_points
                }
            )

        out[int(n_left)] = (
            pd.DataFrame(rows)
            .sort_values("percentile_bin")
            .reset_index(drop=True)
        )

    return out



def aggregate_estimates(estimates: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Loops over estimates from different n_left. First corrects estimates assuming
    geometric distribution correction, by applying

        p_real = 1 - (1 - p_estimate) ** (1 / n_left)

    Then aggregates the corrected estimates and errors, weighting by the number of
    data points used for each estimate.

    Args:
        estimates (Dict[int, pd.DataFrame]): output of get_estimates_of_p_each_n

    Returns:
        pd.DataFrame: with columns 'percentile_bin', 'p_estimate', 'p_error'
    """
    rows = []

    for n_left, df_n in estimates.items():
        required_cols = {"percentile_bin", "p_estimate", "p_error", "n_data_points"}
        missing = required_cols - set(df_n.columns)
        if missing:
            raise ValueError(
                f"DataFrame for n_left={n_left} is missing columns: {sorted(missing)}"
            )

        df_n = df_n.copy()

        # Geometric correction from group-level event probability to
        # per-individual transition probability
        p_group = df_n["p_estimate"].to_numpy(dtype=float)
        p_group = np.clip(p_group, 0.0, 1.0)

        p_real = 1.0 - (1.0 - p_group) ** (1.0 / n_left)

        # Propagate error using derivative of:
        # p_real = 1 - (1-p)^(1/n)
        # dp_real/dp = (1/n) * (1-p)^(1/n - 1)
        deriv = (1.0 / n_left) * (1.0 - p_group) ** (1.0 / n_left - 1.0)
        p_error_real = deriv * df_n["p_error"].to_numpy(dtype=float)

        df_n["p_estimate_real"] = p_real
        df_n["p_error_real"] = p_error_real
        df_n["n_left"] = n_left

        rows.append(df_n)

    if not rows:
        return pd.DataFrame(columns=["percentile_bin", "p_estimate", "p_error"])

    all_estimates = pd.concat(rows, ignore_index=True)

    out_rows = []
    for percentile_bin, subdf in all_estimates.groupby("percentile_bin"):
        weights = subdf["n_data_points"].to_numpy(dtype=float)
        p_vals = subdf["p_estimate_real"].to_numpy(dtype=float)
        err_vals = subdf["p_error_real"].to_numpy(dtype=float)

        valid = np.isfinite(weights) & np.isfinite(p_vals) & np.isfinite(err_vals) & (weights > 0)
        if not np.any(valid):
            out_rows.append(
                {
                    "percentile_bin": percentile_bin,
                    "p_estimate": np.nan,
                    "p_error": np.nan,
                }
            )
            continue

        weights = weights[valid]
        p_vals = p_vals[valid]
        err_vals = err_vals[valid]

        # Weighted mean
        p_estimate = np.average(p_vals, weights=weights)

        # Error on weighted mean from independent component errors
        # Var(sum(w_i x_i)/sum(w_i)) = sum(w_i^2 var_i) / (sum(w_i))^2
        p_error = np.sqrt(np.sum((weights ** 2) * (err_vals ** 2))) / np.sum(weights)

        out_rows.append(
            {
                "percentile_bin": percentile_bin,
                "p_estimate": p_estimate,
                "p_error": p_error,
            }
        )

    return (
        pd.DataFrame(out_rows)
        .sort_values("percentile_bin")
        .reset_index(drop=True)
    )



def estimate_exp_by_percentile_df(
    df: pd.DataFrame,
    percentile_bins: Sequence[float],
    n_boot: int = 100
) -> pd.DataFrame:
    """
    For each percentile window, compute the unbiased exponential rate parameter
    estimate and its error, and return results as a DataFrame.

    Workflow:
        1. Assign each row to a percentile bin using proportion_transitioned
        2. Estimate transition probability separately for each n_left
        3. Aggregate across n_left after geometric correction

    Args:
        df (pd.DataFrame): output of durations.get_transition_duration_table
        percentile_bins (list-like): thresholds of percentile bins for estimation
        n_boot (int): number of bootstrap replicates for SD estimation; passed
            through to get_estimates_of_p_each_n if that function uses it

    Returns:
        pd.DataFrame with columns:
            - percentile_bin: percentile window upper bound
            - p_estimate: unbiased exponential rate parameter estimate
            - p_error: bootstrap standard deviation of the estimate
    """
    required_cols = {"interval_dur", "n_left", "proportion_transitioned"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataframe is missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["percentile_bin"] = assign_percentile_bin(
        df["proportion_transitioned"],
        percentile_bins,
        return_upper=True,
    )

    estimates = get_estimates_of_p_each_n(df, n_boot=n_boot)

    out = aggregate_estimates(estimates)

    return out.sort_values("percentile_bin").reset_index(drop=True)
