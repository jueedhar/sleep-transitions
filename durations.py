# Juee Dhar, 07 Sept 2026
# Pranav Minasandra, 23 Mar 2026
# pminasandra.github.io

"""
Functions for extracting inter-event durations, performance of corrections
"""

from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config
import utilities

EVENTTYPES = ("sleep", "wake")
_COMPLEMENT = {"sleep": "wake", "wake": "sleep"}

def get_intervals(events: np.ndarray) -> np.ndarray:
    """
    Inter-event intervals, sorted earliest to latest. Simultaneous events
    repeat the interval from the last non-simultaneous event.

        [6010, 6023, 6023] -> [13, 13]
        [10, 20, 20, 35]   -> [10, 10, 15]
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

    # repeat each diff once per simultaneous event at that timestamp
    intervals = np.repeat(diffs, counts[1:])

    return intervals


def get_transition_duration_table_edge(events_df, eventtype, group_col="clutch_id", date_col="night_date"):
    """
    Edge events: each animal transitions into `eventtype` once per cohort-night,
    so the at-risk pool (n_left) depletes monotonically.
    """
    if eventtype not in EVENTTYPES:
        raise ValueError("eventtype must be 'sleep' or 'wake'")

    sub = events_df[events_df["event_type"] == eventtype]
    sub = sub.dropna(subset=["event_time", date_col, group_col]).copy()
    if sub.empty:
        return pd.DataFrame()

    cohort = [date_col, group_col]
    sub = sub.sort_values(cohort + ["event_time"]).reset_index(drop=True)

    sub["n_total"] = sub.groupby(cohort)["event_time"].transform("size")
    sub["_rank"] = sub.groupby(cohort)["event_time"].rank(method="dense").astype(int)

    sub = sub[sub.groupby(cohort)["_rank"].transform("max") >= 2].copy()
    if sub.empty:
        return pd.DataFrame()

    bucket = sub.groupby(cohort + ["_rank"]).size().rename("_count").reset_index()
    bucket["_cum_before"] = bucket.groupby(cohort)["_count"].cumsum() - bucket["_count"]

    times = sub.groupby(cohort + ["_rank"])["event_time"].first().reset_index()
    times = times.sort_values(cohort + ["_rank"])
    times["interval_dur"] = ((times["event_time"] - times.groupby(cohort)["event_time"].shift(1))
                             / np.timedelta64(1, "s"))

    meta = bucket.merge(times[cohort + ["_rank", "interval_dur"]], on=cohort + ["_rank"], how="left")
    sub = sub.merge(meta[cohort + ["_rank", "_cum_before", "interval_dur"]],
                    on=cohort + ["_rank"], how="left")

    sub = sub[sub["_rank"] > 1].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["n_left"] = sub["n_total"] - sub["_cum_before"]
    sub["proportion_transitioned"] = sub["_cum_before"] / sub["n_total"]
    sub["eventtype"] = eventtype
    return sub.drop(columns=["_rank", "_cum_before"]).reset_index(drop=True)


def get_transition_duration_tables_bulk(events_df, group_col="clutch_id", date_col="night_date"):
    """
    Bulk (in-night) events, both eventtypes in one per-animal state pass.
    Animals transition repeatedly, so the at-risk pool (n_left, aka M_r) is
    the live count of tracked animals in the complementary state, not a
    monotonically shrinking pool. Each animal starts being tracked at its
    first event, seeded to that event's complement state.

    Same-timestamp events are batched, as in get_transition_duration_table_edge,
    so interval_dur is always a gap between distinct times. Sleep/wake edge
    exclusion is independent per type, so a retained event can repeat an
    animal's known state (the opposite-type flip between them got excluded);
    such repeats aren't real transitions and are dropped.

    Returns:
        {"sleep": table, "wake": table}
    """
    df = events_df.dropna(subset=["event_time", date_col, group_col, "animal_id", "event_type"]).copy()
    if df.empty:
        return {eventtype: pd.DataFrame() for eventtype in EVENTTYPES}

    cohort = [date_col, group_col]
    df = df.sort_values(cohort + ["event_time"])

    rows = {eventtype: [] for eventtype in EVENTTYPES}
    for _, cohort_df in tqdm(df.groupby(cohort, sort=False), desc="bulk duration tables"):
        state = {}
        occupancy = {"sleep": 0, "wake": 0}
        prev_time = {"sleep": None, "wake": None}

        for event_time, batch in cohort_df.groupby("event_time", sort=False):
            for animal, etype in zip(batch["animal_id"], batch["event_type"]):
                if animal not in state:
                    prior = _COMPLEMENT[etype]
                    state[animal] = prior
                    occupancy[prior] += 1

            genuine_mask = [state[a] != e for a, e in zip(batch["animal_id"], batch["event_type"])]
            genuine = batch[genuine_mask]

            n_total = len(state)
            for eventtype in EVENTTYPES:
                transitioners = genuine[genuine["event_type"] == eventtype]
                if transitioners.empty:
                    continue
                if prev_time[eventtype] is not None:
                    n_left = occupancy[_COMPLEMENT[eventtype]]
                    proportion_transitioned = occupancy[eventtype] / n_total
                    interval_dur = (event_time - prev_time[eventtype]) / np.timedelta64(1, "s")
                    for _, row in transitioners.iterrows():
                        row_out = row.to_dict()
                        row_out.update({
                            "n_total": n_total,
                            "n_left": n_left,
                            "proportion_transitioned": proportion_transitioned,
                            "interval_dur": interval_dur,
                            "eventtype": eventtype,
                        })
                        rows[eventtype].append(row_out)
                prev_time[eventtype] = event_time

            for animal, etype in zip(genuine["animal_id"], genuine["event_type"]):
                occupancy[state[animal]] -= 1
                state[animal] = etype
                occupancy[etype] += 1

    return {eventtype: pd.DataFrame(rows[eventtype]).reset_index(drop=True) for eventtype in EVENTTYPES}


def get_transition_duration_table(df: pd.DataFrame, eventtype: str) -> pd.DataFrame:
    """
    Per date x clutch_id: inter-event durations for sleep/wake transitions,
    one row per transitioning individual (simultaneous transitioners share
    the same interval_dur). n_left is the number still not transitioned; the
    first transition per subset has no preceding interval and is skipped.
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
""" 
if __name__ == "__main__":
    focaldate = pd.to_datetime("2025-01-01").date()
    masterdf = pd.read_parquet(config.MASTER_DATA_SHEET)
    masterdf.dropna(inplace=True)

    t_df = get_transition_duration_table(masterdf, "wake")
    print(t_df)
"""

