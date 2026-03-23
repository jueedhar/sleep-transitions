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


def get_transition_duration_table(df: pd.DataFrame, eventtype: str) -> List[np.ndarray]:
    """
    Loops across dates, and within each date across clutch_ids, and returns a list
    of inter-event durations for sleep or wake events.

    Each element of the returned list corresponds to one date × clutch_id subset
    and contains the consecutive intervals between ordered events in that subset.

    Args:
        df (pd.DataFrame): master data frame with all sleep/wake data
        eventtype (str): "sleep" or "wake"

    Returns:
        List[np.ndarray]: list of arrays of inter-event durations
    """
    if eventtype not in {"sleep", "wake"}:
        raise ValueError("eventtype must be either 'sleep' or 'wake'")

    timecol = f"t_{eventtype}"
    if timecol not in df.columns:
        raise ValueError(f"Column '{timecol}' not found in dataframe")

    out: List[np.ndarray] = []

    for (_, _), subdf in df.groupby(["date", "clutch_id"]):
        events = subdf[timecol].dropna().to_numpy()
        intervals = get_intervals(events)

        if intervals.size > 0:
            out.append(intervals)

    return out

