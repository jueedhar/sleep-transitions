# Pranav Minasandra
# 23 Mar 2026
# pminasandra.github.io

from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
import durations
import estimation
import simulations
import utilities


def f_memoryless(n: int, n_total: int) -> float:
    """Constant transition probability."""
    return 0.003


def f_increasing(n: int, n_total: int) -> float:
    """Quadratically increasing transition probability."""
    return 0.001 + 0.009 * (n / n_total) ** 2


def f_saturating(n: int, n_total: int) -> float:
    """Rapidly saturating transition probability."""
    return 0.001 + 0.009 * (n / n_total) ** 0.2


def run_transition_recovery_benchmark(
    n_reps: int = 100,
    n_boot: int = 100,
    t_max: int = 5000,
    n_total_sim: int = 80,
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    Run repeated simulations for the three matched transition models
    (memoryless, increasing, saturating), estimate exponential parameters by
    percentile separately for wake and sleep transitions, and plot the true input
    function against the recovered estimates.

    Uses:
        - simulations.simulate_master_df()
        - durations.get_transition_duration_table()
        - estimation.accumulate_percentile_window_durations()
        - estimation.estimate_exp_by_percentile_df()

    Assumes the following are defined in config.py:
        - config.PERCENTILE_THRESHOLDS
        - config.MIN_TAGS

    Args:
        n_reps (int): number of simulated datasets per scenario
        n_boot (int): bootstrap replicates passed to
            estimation.estimate_exp_by_percentile_df()
        t_max (int): maximum number of time steps per simulation
        n_total_sim (int): total number of individuals per simulated clutch

    Returns:
        Tuple containing:
            - all_estimates (pd.DataFrame): one row per replicate × percentile × event
            - summary (pd.DataFrame): mean estimate and mean bootstrap SD by scenario,
              event, and percentile
            - fig (plt.Figure): 3 × 2 panel plot
    """

    scenario_functions: Dict[str, Callable[[int, int], float]] = {
        "memoryless": f_memoryless,
        "increasing": f_increasing,
        "saturating": f_saturating,
    }

    estimate_frames = []

    for scenario_name, f_transition in scenario_functions.items():
        print(scenario_name)
        for rep in range(n_reps):
            print(f"sim {rep} of {n_reps}")
            df_sim_main = simulations.simulate_master_df(
                f_wake_transition_dependence=f_transition,
                f_sleep_transition_dependence=f_transition,
                t_max=t_max,
                n_total=n_total_sim,
            )


            for clutch_id, df_sim in df_sim_main.groupby("clutch_id"):
                for eventtype in ["wake", "sleep"]:
                    df_sim = df_sim[df_sim[f't_{eventtype}'] > 0]
                    percentile_durations = estimation.accumulate_percentile_window_durations(
                        df = df_sim,
                        eventtype = eventtype,
                        perc_thresholds=config.PERCENTILE_THRESHOLDS,
                        min_tags=config.MIN_TAGS,
                    )

                    est_df = estimation.estimate_exp_by_percentile_df(
                        percentile_durations=percentile_durations,
                        n_boot=n_boot,
                    )

                    est_df["scenario"] = scenario_name
                    est_df["eventtype"] = eventtype
                    est_df["rep"] = rep
                    est_df["clutch_id"] = clutch_id
                    estimate_frames.append(est_df)

    all_estimates = pd.concat(estimate_frames, ignore_index=True)

    summary = (
        all_estimates
        .groupby(["scenario", "eventtype", "percentile", "clutch_id"], as_index=False)
        .agg(
            estimate_mean=("estimate", "mean"),
            estimate_sd_across_reps=("estimate", "std"),
            bootstrap_sd_mean=("sd", "mean"),
            bootstrap_sd_sd=("sd", "std"),
            n_mean=("n", "mean"),
        )
    )

    # Truth curves: evaluate each input function at n = 0..100, normalized to [0, 1]
    truth_frames = []
    for scenario_name, f_transition in scenario_functions.items():
        truth_frames.append(
            pd.DataFrame(
                {
                    "scenario": scenario_name,
                    "n": list(range(101)),
                    "x": [n / 100 for n in range(101)],
                    "true_prob": [f_transition(n, 100) for n in range(101)],
                }
            )
        )
    truth_df = pd.concat(truth_frames, ignore_index=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True
    )

    scenario_order = ["memoryless", "increasing", "saturating"]
    event_order = ["wake", "sleep"]

    for i, scenario_name in enumerate(scenario_order):
        for j, eventtype in enumerate(event_order):
            ax = axes[i, j]

            truth_sub = truth_df[truth_df["scenario"] == scenario_name]
            est_sub = summary[
                (summary["scenario"] == scenario_name)
                & (summary["eventtype"] == eventtype)
            ].sort_values("percentile")

            sns.lineplot(
                data=truth_sub,
                x="x",
                y="true_prob",
                ax=ax,
                color="black",
            )

            for clutch_id, est_sub_cl in est_sub.groupby("clutch_id"):
                sns.lineplot(
                    data=est_sub_cl,
                    x="percentile",
                    y="estimate_mean",
                    label=f"estimate: {clutch_id}",
                    marker="o",
                    ax=ax,
                )

                ax.errorbar(
                    est_sub["percentile"],
                    est_sub["estimate_mean"],
                    yerr=est_sub["bootstrap_sd_mean"],
                    fmt="none",
                    capsize=3,
                    color=sns.color_palette()[0],
                )

            if i == 0:
                ax.set_title(eventtype)
            if j == 0:
                ax.set_ylabel(f"{scenario_name}\nprobability / estimate")
            else:
                ax.set_ylabel("")

            ax.set_xlabel("fraction transitioned")
            ax.set_xlim(0, 1)

    return all_estimates, summary, fig

if __name__ == "__main__":
    all_estimates, summary, fig = run_transition_recovery_benchmark(
        n_reps=10,
        n_boot=100,
        t_max=5000,
        n_total_sim=80)

    print(summary)
    utilities.saveimg(fig, "sim_results")
