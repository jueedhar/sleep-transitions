# Pranav Minasandra
# 26 Mar 2026
# pminasandra.github.io

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
import durations
import estimation
import visualisation
import utilities

def get_percentile_transition_estimates(
    df: pd.DataFrame,
    eventtype: str,
    percentile_bins = config.PERCENTILE_THRESHOLDS,
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

if __name__ == "__main__":
    masterdf = pd.read_parquet(config.MASTER_DATA_SHEET)
    results = {}
    for state in ("sleep", "wake"):
        results[state] = get_percentile_transition_estimates(masterdf, state)

    fig, axs = plt.subplots(1, 2, sharey=True)
    i = 0
    for state in results:
        ax = axs[i]

        df = results[state]
        sns.lineplot(data=df, x='percentile_bin', y='p_estimate',
                            linewidth=0.5, ax=ax)
        ax.errorbar(x=df['percentile_bin'], y=df['p_estimate'], yerr=df['p_error'],
                        fmt='none', color='black', linewidth=0.5)

        ax.set_title(state)
        ax.set_xlabel(f"Proportion a{state}")
        ax.set_ylabel("Transition probability")
        i+=1

    utilities.saveimg(fig, "real_transition_probabilities")
