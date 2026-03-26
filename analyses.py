# Pranav Minasandra
# 26 Mar 2026
# pminasandra.github.io

import pandas as pd

import durations
import estimation

def get_percentile_transition_estimates(
    df: pd.DataFrame,
    eventtype: str,
    percentile_bins,
    n_boot: int = 100,
) -> pd.DataFrame:
    """
    From the master dataframe, compute percentile-wise transition probability
    estimates for one event type.

    Workflow:
        1. Build the transition-duration table with
           durations.get_transition_duration_table()
        2. Estimate percentile-wise probabilities with
           estimation.estimate_exp_by_percentile_df()

    Args:
        df (pd.DataFrame): master dataframe
        eventtype (str): "sleep" or "wake"
        percentile_bins (list-like): thresholds of percentile bins for estimation
        n_boot (int): number of bootstrap replicates passed to
            estimation.estimate_exp_by_percentile_df()

    Returns:
        pd.DataFrame: final percentile table with columns:
            - percentile_bin
            - p_estimate
            - p_error
    """
    transition_df = durations.get_transition_duration_table(df, eventtype=eventtype)

    if transition_df.empty:
        return pd.DataFrame(
            columns=["percentile_bin", "p_estimate", "p_error"]
        )

    out = estimation.estimate_exp_by_percentile_df(
        df=transition_df,
        percentile_bins=percentile_bins,
        n_boot=n_boot,
    )

    return out.sort_values("percentile_bin").reset_index(drop=True)
