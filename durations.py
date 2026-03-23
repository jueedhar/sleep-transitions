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
    Given a list containing the time of day (in seconds 0 <= t < 86400), returns the
    consecutive inter-event durations starting from the first to the last.

    Args:
        events (np.ndarray): time-stamps (in seconds) of all events

    Returns:
        np.ndarray: consecutive inter-event intervals
    """
    if events.size == 0:
        return np.array([])

    # Ensure sorted order
    events_sorted = np.sort(events)

    # Compute consecutive differences
    intervals = np.diff(events_sorted)

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

