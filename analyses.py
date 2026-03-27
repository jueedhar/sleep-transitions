# Pranav Minasandra
# 26 Mar 2026
# pminasandra.github.io

from typing import Sequence
import math

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
    percentile_bins=config.PERCENTILE_THRESHOLDS,
    n_boot: int = 100,
    foreach: str = "none",
) -> pd.DataFrame:
    """
    From the master dataframe, compute percentile-wise transition probability
    estimates for one event type.

    Workflow:
        1. Build the transition-duration table with
           durations.get_transition_duration_table()
        2. Estimate percentile-wise probabilities with
           estimation.estimate_exp_by_percentile_df()

    If foreach != "none", estimates are returned separately for each unique value
    in that column.

    Args:
        df (pd.DataFrame): master dataframe
        eventtype (str): "sleep" or "wake"
        percentile_bins (list-like): thresholds of percentile bins for estimation
        n_boot (int): number of bootstrap replicates passed to
            estimation.estimate_exp_by_percentile_df()
        foreach (str): either "none", or a column name in df along which estimates
            should be computed separately

    Returns:
        pd.DataFrame: final percentile table with columns:
            - percentile_bin
            - p_estimate
            - p_error
        and, if foreach != "none":
            - foreach
    """
    transition_df = durations.get_transition_duration_table(df, eventtype=eventtype)

    if transition_df.empty:
        cols = ["percentile_bin", "p_estimate", "p_error"]
        if foreach != "none":
            cols = [foreach] + cols
        return pd.DataFrame(columns=cols)

    out = estimation.estimate_exp_by_percentile_df(
        df=transition_df,
        percentile_bins=percentile_bins,
        n_boot=n_boot,
        foreach=foreach,
    )

    sort_cols = ["percentile_bin"] if foreach == "none" else [foreach, "percentile_bin"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def analyse_sleep_wake_asymmetry_by(
    masterdf: pd.DataFrame,
    by: str = "none",
    foreach: str = "none",
    drop_vals: Sequence = ("Unknown",),
):
    """
    Analyses sleep-wake asymmetries by multiple variables. Returns datatable and fig,
    ax.

    Plotting behaviour:
        - If by == "none" and foreach == "none":
            one panel containing both sleep and wake curves
        - Otherwise:
            two panels, one for sleep and one for wake, with separate lines for each
            unique combination of `by` and `foreach`

    Args:
        masterdf (pd.DataFrame): main input read
        by (str): column label, something that is constant within a given date-clutch,
            e.g., group_sleep_site. (default 'none')
        foreach (str): column label, something that varies within a given date-clutch,
            e.g., age, or sex
        drop_vals (Sequence): values of masterdf[by] or masterdf[foreach] that need to
            be dropped.

    Returns:
        pd.DataFrame: with columns percentile_bin, p_estimate, p_error, by, foreach,
            eventtype, and label.
        plt.Figure
        plt.Axes or np.ndarray of Axes
    """
    mdf = masterdf.copy()

    if by != "none" and by not in mdf.columns:
        raise ValueError(f"`by` = '{by}' is not a column in masterdf")

    if foreach != "none" and foreach not in mdf.columns:
        raise ValueError(f"`foreach` = '{foreach}' is not a column in masterdf")

    if by != "none":
        mdf = mdf[~mdf[by].isin(drop_vals)]

    if foreach != "none":
        mdf = mdf[~mdf[foreach].isin(drop_vals)]

    result_frames = []

    # Build analysis groups
    if by == "none" and foreach == "none":
        groups = [("all", mdf)]
    elif by != "none" and foreach == "none":
        groups = [(str(by_val), dfs) for by_val, dfs in mdf.groupby(by)]
    elif by == "none" and foreach != "none":
        groups = [(str(foreach_val), dfs) for foreach_val, dfs in mdf.groupby(foreach)]
    else:
        groups = [
            (f"{by_val} | {foreach_val}", dfs)
            for (by_val, foreach_val), dfs in mdf.groupby([by, foreach])
        ]

    for label, dfs in groups:
        for eventtype in ["sleep", "wake"]:
            res = get_percentile_transition_estimates(
                dfs,
                eventtype=eventtype,
                foreach="none",
            )

            if res.empty:
                continue

            res = res.copy()
            res["eventtype"] = eventtype
            res["label"] = label

            if by != "none":
                if foreach == "none":
                    # label here is just the by value
                    res[by] = label
                else:
                    # recover from group subset directly
                    vals = dfs[by].dropna().unique()
                    res[by] = vals[0] if len(vals) == 1 else pd.NA

            if foreach != "none":
                if by == "none":
                    res[foreach] = label
                else:
                    vals = dfs[foreach].dropna().unique()
                    res[foreach] = vals[0] if len(vals) == 1 else pd.NA

            result_frames.append(res)

    out_cols = ["percentile_bin", "p_estimate", "p_error"]
    if by != "none":
        out_cols.append(by)
    if foreach != "none":
        out_cols.append(foreach)
    out_cols += ["eventtype", "label"]

    if not result_frames:
        fig, ax = plt.subplots()
        return pd.DataFrame(columns=out_cols), fig, ax

    out = pd.concat(result_frames, ignore_index=True)

    sns.set_theme(style="whitegrid")

    if by == "none" and foreach == "none":
        fig, ax = plt.subplots(figsize=(6, 4))

        for eventtype in ["sleep", "wake"]:
            plotdf = (
                out[out["eventtype"] == eventtype]
                .sort_values("percentile_bin")
            )

            sns.lineplot(
                data=plotdf,
                x="percentile_bin",
                y="p_estimate",
                marker="o",
                linewidth=0.7,
                ax=ax,
                label=eventtype,
                errorbar=None,
            )

            ax.errorbar(
                plotdf["percentile_bin"],
                plotdf["p_estimate"],
                yerr=plotdf["p_error"],
                fmt="none",
                capsize=2,
                linewidth=0.6,
            )

        ax.set_title("sleep and wake")
        ax.set_xlabel("percentile_bin")
        ax.set_ylabel("p_estimate")
        ax.legend(fontsize=8, title_fontsize=9, frameon=False)
        fig.tight_layout()
        return out[out_cols], fig, ax

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    event_axes = {"sleep": axes[0], "wake": axes[1]}

    # stable colors by label
    labels = list(pd.unique(out["label"]))
    palette = sns.color_palette(n_colors=max(len(labels), 1))
    color_map = dict(zip(labels, palette))

    for eventtype in ["sleep", "wake"]:
        ax = event_axes[eventtype]
        event_df = out[out["eventtype"] == eventtype]

        for label, subdf in event_df.groupby("label"):
            plotdf = subdf.sort_values("percentile_bin")
            color = color_map[label]

            sns.lineplot(
                data=plotdf,
                x="percentile_bin",
                y="p_estimate",
                marker="o",
                linewidth=0.7,
                ax=ax,
                label=label,
                color=color,
                errorbar=None,
            )

            ax.errorbar(
                plotdf["percentile_bin"],
                plotdf["p_estimate"],
                yerr=plotdf["p_error"],
                fmt="none",
                color=color,
                capsize=2,
                linewidth=0.6,
            )

        ax.set_title(eventtype)
        ax.set_xlabel("percentile_bin")
        ax.set_ylabel("p_estimate")
        ax.legend(fontsize=8, title_fontsize=9, frameon=False)

    fig.tight_layout()
    return out[out_cols], fig, axes


if __name__ == "__main__":
    masterdf = pd.read_parquet(config.MASTER_DATA_SHEET)
    masterdf.dropna(inplace=True)

    analyse_sleep_wake_asymmetry_by(masterdf)
