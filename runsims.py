# Pranav Minasandra
# 23 Mar 2026
# pminasandra.github.io

from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import analyses
import config
import simulations
import utilities

def int_to_color(i: int, cmap_name: str = "tab20"):
    """
    Map an integer to a deterministic matplotlib color.

    Args:
        i (int): input integer
        cmap_name (str): name of matplotlib colormap (default: 'tab20')

    Returns:
        tuple: RGBA color usable in matplotlib
    """
    cmap = plt.get_cmap(cmap_name)
    n = cmap.N  # number of discrete colors in the cmap

    # ensure deterministic wrap
    idx = int(i) % n

    return cmap(idx)

def f_memoryless(n, n_total: int) -> float:
    """Constant transition probability."""
    if isinstance(n, int):
        return 0.003
    else:
        return np.array([0.003]*len(n))


def f_increasing(n, n_total: int) -> float:
    """Quadratically increasing transition probability."""
    return 0.001 + 0.009 * (n / n_total) ** 2


def f_saturating(n, n_total: int) -> float:
    """Rapidly saturating transition probability."""
    return 0.001 + 0.009 * (n / n_total) ** 0.2


def run_transition_recovery_benchmark(
    n_reps: int = 100,
    n_boot: int = 100,
    t_max: int = 5000,
    n_total_sim: int = 80,
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    """

    scenario_functions: Dict[str, Callable[[int, int], float]] = {
        "memoryless": f_memoryless,
        "increasing": f_increasing,
        "saturating": f_saturating,
    }

    estimate_frames = []

    fig, axs = plt.subplots(len(scenario_functions), 2, sharey=True)
    row = 0
    for scenario_name, f_transition in scenario_functions.items():
        print(scenario_name)
        first_rep = True
        for rep in range(n_reps):
            print(f"sim {rep+1} of {n_reps}")
            df_sim_main = simulations.simulate_master_df(
                f_wake_transition_dependence=f_transition,
                f_sleep_transition_dependence=f_transition,
                t_max=t_max,
                n_total=n_total_sim,
            )


            for clutch_id, df_sim_sub in df_sim_main.groupby("clutch_id"):
                clutch_size = int(clutch_id.split('_')[1])
                for eventtype in ("sleep", "wake"):
                    col = 0 if eventtype == "sleep" else 1
                    ax = axs[row, col]
                    if row == 0:
                        ax.set_title(eventtype)
                    if col == 0:
                        ax.set_ylabel(scenario_name)

                    if first_rep:
                        x = np.arange(1, 101)
                        y = f_transition(x, 100)
                        label = "True relation" if clutch_size == 20 else None
                        ax.plot(x/100, y, color="black", linewidth=0.8, label=label)


                    perc_results = analyses.get_percentile_transition_estimates(
                                        df_sim_sub,
                                        eventtype,
                                        percentile_bins=config.PERCENTILE_THRESHOLDS
                                        )
                    color = int_to_color(clutch_size, cmap_name='Set1')
                    label = clutch_id if first_rep else None
                    sns.lineplot(data=perc_results, x="percentile_bin", y="p_estimate",
                                        ax=ax, linewidth=0.3, color=color, label=label,
                                        alpha=0.3)
                    ax.errorbar(x=perc_results["percentile_bin"],
                                    y=perc_results["p_estimate"],
                                    yerr=perc_results["p_error"],
                                    fmt='none',
                                    linewidth=0.2,
                                    color=color,
                                    alpha=0.3)
                    ax.legend(
                            fontsize=6,
                            title_fontsize=6,
                            markerscale=0.6,
                            handlelength=1,
                            labelspacing=0.3,   # vertical spacing
                            borderpad=0.3       # padding inside legend box
                        )
            first_rep = False


        row += 1
        print()
    return fig


if __name__ == "__main__":
    fig = run_transition_recovery_benchmark(
        n_reps=25,
        n_boot=100,
        t_max=5000,
        n_total_sim=80)

    utilities.saveimg(fig, "sim_results")
