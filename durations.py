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
        return np.array([], dtype=events.dtype)

    diffs = np.diff(unique_times)

    # Repeat each diff according to how many events occur at the later timestamp
    intervals = np.repeat(diffs, counts[1:])

    return intervals


def get_transition_duration_table(df: pd.DataFrame, eventtype: str) -> pd.DataFrame:
    """
    Loops across dates, and within each date across clutch_ids, and returns a table
    of inter-event durations for sleep or wake events.

    Each output row corresponds to one transition interval. If multiple individuals
    transition simultaneously at the later timestamp, that interval is repeated once
    per individual, matching the behaviour of get_intervals().

    For each interval row:
        - interval_dur: time until the focal transition
        - n_total: total number of observed individuals in that date × clutch subset
        - n_left: number of individuals still not transitioned during that interval
        - proportion_transitioned: proportion already transitioned during that
          interval, i.e. before the focal transition

    Metadata columns from the master dataframe are copied in. For columns that are
    not constant within a date × clutch subset, NaN is inserted.

    Args:
        df (pd.DataFrame): master data frame with all sleep/wake data
        eventtype (str): "sleep" or "wake"

    Returns:
        pd.DataFrame: with columns 'interval_dur', 'n_total', 'n_left',
        'proportion_transitioned', and metadata columns from the master dataframe.
    """
    if eventtype not in {"sleep", "wake"}:
        raise ValueError("eventtype must be either 'sleep' or 'wake'")

    timecol = f"t_{eventtype}"
    if timecol not in df.columns:
        raise ValueError(f"Column '{timecol}' not found in dataframe")

    rows = []

    # These columns are individual-specific event data and do not map cleanly onto
    # interval rows, so they are excluded from direct copying.
    excluded_metadata = {"ind", "t_wake", "t_sleep"}
    metadata_cols = [c for c in df.columns if c not in excluded_metadata]

    for (_, _), subdf in df.groupby(["date", "clutch_id"]):
        events = subdf[timecol].dropna().to_numpy()
        if events.size < 2:
            continue

        events_sorted = np.sort(events)
        intervals = get_intervals(events_sorted)

        if intervals.size == 0:
            continue

        unique_times, counts = np.unique(events_sorted, return_counts=True)
        diffs = np.diff(unique_times)

        # Build subset-level metadata: keep constant values, else NaN
        metadata = {}
        for col in metadata_cols:
            vals = subdf[col].dropna().unique()
            metadata[col] = vals[0] if len(vals) == 1 else np.nan

        n_total = len(events_sorted)
        transitioned_so_far = counts[0]

        for diff, later_count in zip(diffs, counts[1:]):
            n_left = n_total - transitioned_so_far
            proportion_transitioned = transitioned_so_far / n_total

            for _ in range(later_count):
                row = {
                    "interval_dur": diff,
                    "n_total": n_total,
                    "n_left": n_left,
                    "proportion_transitioned": proportion_transitioned,
                    "eventtype": eventtype,
                }
                row.update(metadata)
                rows.append(row)

            transitioned_so_far += later_count

    return pd.DataFrame(rows)

if __name__ == "__main__":
    focaldate = pd.to_datetime("2025-01-01").date()
    masterdf = pd.read_parquet(config.MASTER_DATA_SHEET)
    masterdf.dropna(inplace=True)

    t_df = get_transition_duration_table(masterdf, "wake")


