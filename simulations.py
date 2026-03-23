# Pranav Minasandra
# 23 Mar 2026
# pminasandra.github.io

"""
Simulates wake and sleep data table for testing.
"""

from typing import Callable, Dict, List
import uuid

import numpy as np
import pandas as pd


def simulate_transition_times(
    n_total: int,
    f_transition_dependence: Callable[[int], float],
    t_max: int = 5000,
    seed_one_awake: bool = True,
) -> np.ndarray:
    """
    Simulate irreversible A -> B transitions for all agents and return the transition
    time of every agent.

    Args:
        n_total (int): total number of agents
        f_transition_dependence (Callable[[int], float]): maps the current number of
            agents already in state B to the per-agent probability of switching on
            the current time step
        t_max (int): maximum number of time steps to simulate
        seed_one_awake (bool): if True, initialise agent 0 in state B at time 0

    Returns:
        np.ndarray: length-n_total array of transition times. Agents that do not
        switch by t_max are assigned -1. If seed_one_awake is True, agent 0 has
        transition time 0.
    """
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    if t_max < 0:
        raise ValueError("t_max must be non-negative")

    in_b = np.zeros(n_total, dtype=bool)
    transition_times = np.full(n_total, -1, dtype=int)

    if seed_one_awake:
        in_b[0] = True
        transition_times[0] = 0

    for t in range(t_max):
        n_in_b = int(in_b.sum())
        if n_in_b == n_total:
            break

        p = f_transition_dependence(n_in_b, n_total)
        if not (0.0 <= p <= 1.0):
            raise ValueError("f_transition_dependence must return a value in [0, 1]")

        candidates = ~in_b
        draws = np.random.random(n_total) < p
        switching = candidates & draws

        in_b[switching] = True
        transition_times[switching] = t

    return transition_times


def simulate_master_df(
    f_wake_transition_dependence: Callable[[int, int], float],
    f_sleep_transition_dependence: Callable[[int, int], float],
    t_max: int = 5000,
    n_total: int = 80
) -> pd.DataFrame:
    """
    Simulate 5 days of waking and sleeping transitions for 5 clutches and return the
    result in the master-dataframe format.

    Design:
    - Dates: 2020-11-28 through 2020-12-02 inclusive
    - 5 clutches per day
    - Each clutch has 80 total individuals
    - Sample sizes per clutch are 20, 30, 40, 50, 60
    - Individual IDs are random UUID strings, but fixed within each clutch across days
    - Wake times are shifted by 6 * 3600 seconds
    - Sleep times are shifted by 18 * 3600 seconds
    - All other columns are filled with NaN for now

    Args:
        f_wake_transition_dependence (Callable[[int, int], float]): transition-probability
            function for waking
        f_sleep_transition_dependence (Callable[[int, int], float]): transition-probability
            function for sleeping
        t_max (int): maximum number of simulation steps for each transition process
        n_total (int): num of individuals in simulated group

    Returns:
        pd.DataFrame: simulated dataframe with columns
            [date, ind, t_wake, t_sleep, group_id, clutch_id, sleep_site_type,
             wake_site_type, age, sex, group_size]
    """
    start_date = pd.Timestamp("2020-11-28")
    dates = pd.date_range(start_date, periods=30, freq="D")

#    clutch_sample_sizes = [20, 30, 40, 50, 60]
    clutch_sample_sizes = [80, 80, 80, 80, 80]
    wake_offset = 6 * 3600
    sleep_offset = 18 * 3600

    # Stable clutch IDs and stable pools of individual IDs within clutch
    clutch_ids = [f"clutch_{n_needed}" for n_needed in clutch_sample_sizes]
    clutch_individuals: Dict[str, List[str]] = {
        clutch_id: [str(uuid.uuid4()) for _ in range(n_total)]
        for clutch_id in clutch_ids
    }

    rows = []

    for date in dates:
        for clutch_id, n_needed in zip(clutch_ids, clutch_sample_sizes):
            # Simulate full transition times for all 80 individuals
            wake_times_all = simulate_transition_times(
                n_total=n_total,
                f_transition_dependence=f_wake_transition_dependence,
                t_max=t_max,
                seed_one_awake=True,
            )
            sleep_times_all = simulate_transition_times(
                n_total=n_total,
                f_transition_dependence=f_sleep_transition_dependence,
                t_max=t_max,
                seed_one_awake=True,
            )

            # Sample the observed individuals for this clutch-day once, and use the
            # same sampled identities for both wake and sleep
            sampled_idx = np.random.choice(n_total, size=n_needed, replace=False)

            for idx in sampled_idx:
                t_wake = wake_times_all[idx]
                t_sleep = sleep_times_all[idx]

                rows.append(
                    {
                        "date": date,
                        "ind": clutch_individuals[clutch_id][idx],
                        "t_wake": np.nan if t_wake < 0 else wake_offset + t_wake,
                        "t_sleep": np.nan if t_sleep < 0 else sleep_offset + t_sleep,
                        "group_id": np.nan,
                        "clutch_id": clutch_id,
                        "sleep_site_type": np.nan,
                        "wake_site_type": np.nan,
                        "age": np.nan,
                        "sex": np.nan,
                        "group_size": np.nan,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "ind",
            "t_wake",
            "t_sleep",
            "group_id",
            "clutch_id",
            "sleep_site_type",
            "wake_site_type",
            "age",
            "sex",
            "group_size",
        ],
    )

if __name__ == "__main__":
    def f_wake(n, n_total):
        return 0.005
    def f_sleep(n, n_total):
        return 0.003

    df_sim = simulate_master_df(f_wake, f_sleep)
    print(df_sim)
