# Pranav Minasandra
# 23 Mar 2026
# pminasandra.github.io

"""
Functions for extracting inter-event durations, performance of corrections
"""

from typing import List

import numpy as np
import pandas as pd

import config
import utilities

def get_intervals(events: np.ndarray) -> np.ndarray:
    """
    Given event times within a day (in seconds, 0 <= t < 86400), return the
    consecutive inter-event intervals after ordering events from earliest to latest.

    If multiple events are simultaneous, they are treated as occurring at the same
    transition point. The interval from the previous non-simultaneous event is
    repeated for each simultaneous event.

    Example:
        [6010, 6023, 6023] -> [13, 13]
        [10, 20, 20, 35]   -> [10, 10, 15]

    Args:
        events (np.ndarray): 1D array of event times.

    Returns:
        np.ndarray: consecutive inter-event intervals between non-simultaneous events.
    """
    events = np.asarray(events)

    if events.size < 2:
        return np.array([], dtype=events.dtype)

    events_sorted = np.sort(events)
    unique_times, counts = np.unique(events_sorted, return_counts=True)

    if unique_times.size < 2:
        return np.array([], dtype=float)

    diffs = np.diff(unique_times)
    if diffs.dtype == np.dtype("timedelta64[ns]") or diffs.dtype == np.timedelta64:
        diffs = diffs.astype(float)
        diffs /= int(1_000_000_000)

    # Repeat each diff according to how many events occur at the later timestamp
    intervals = np.repeat(diffs, counts[1:])

    return intervals


def get_transition_duration_table(df: pd.DataFrame, eventtype: str) -> pd.DataFrame:
    """
    Loops across dates, and within each date across clutch_ids, and returns a table
    of inter-event durations for sleep or wake events.

    Each output row corresponds to a specific individual transition, not just an
    abstract repeated interval. If multiple individuals transition simultaneously at
    the later timestamp, one row is emitted per transitioning individual, each carrying
    that individual's metadata.

    For each interval row:
        - interval_dur: time until the focal transition
        - n_total: total number of observed individuals in that date × clutch subset
        - n_left: number of individuals still not transitioned during that interval
        - proportion_transitioned: proportion already transitioned during that
          interval, i.e. before the focal transition

    The first transition timestamp in each date × clutch subset is skipped, as there
    is no preceding inter-event duration to assign to it.

    Args:
        df (pd.DataFrame): master data frame with all sleep/wake data
        eventtype (str): "sleep" or "wake"

    Returns:
        pd.DataFrame: with columns 'interval_dur', 'n_total', 'n_left',
        'proportion_transitioned', and all original columns from the master dataframe.
    """
    if eventtype not in {"sleep", "wake"}:
        raise ValueError("eventtype must be either 'sleep' or 'wake'")

    timecol = f"t_{eventtype}"
    if timecol not in df.columns:
        raise ValueError(f"Column '{timecol}' not found in dataframe")

    rows = []

    for (_, _), subdf in df.groupby(["date", "clutch_id"]):
        subdf_event = subdf.dropna(subset=[timecol]).copy()
        if len(subdf_event) < 2:
            continue

        subdf_event = subdf_event.sort_values(timecol)

        times = subdf_event[timecol].to_numpy()
        unique_times, counts = np.unique(times, return_counts=True)

        if len(unique_times) < 2:
            continue

        diffs = np.diff(unique_times)
        if np.issubdtype(diffs.dtype, np.timedelta64):
            diffs = diffs / np.timedelta64(1, "s")

        n_total = len(subdf_event)
        transitioned_so_far = counts[0]

        for prev_time, this_time, diff, later_count in zip(
            unique_times[:-1], unique_times[1:], diffs, counts[1:]
        ):
            n_left = n_total - transitioned_so_far
            proportion_transitioned = transitioned_so_far / n_total

            transitioners = subdf_event[subdf_event[timecol] == this_time].copy()

            # Safety check: this should match later_count from np.unique
            if len(transitioners) != later_count:
                raise RuntimeError(
                    "Mismatch between unique-count calculation and transitioning rows"
                )

            for _, ind_row in transitioners.iterrows():
                row = ind_row.to_dict()
                row.update(
                    {
                        "interval_dur": float(diff),
                        "n_total": n_total,
                        "n_left": n_left,
                        "proportion_transitioned": proportion_transitioned,
                        "eventtype": eventtype,
                    }
                )
                rows.append(row)

            transitioned_so_far += later_count

    return pd.DataFrame(rows)

if __name__ == "__main__":
    focaldate = pd.to_datetime("2025-01-01").date()
    masterdf = pd.read_parquet(config.MASTER_DATA_SHEET)
    masterdf.dropna(inplace=True)

    t_df = get_transition_duration_table(masterdf, "wake")
    print(t_df)


