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
BULK_MAX_INTERVAL_DUR_SEC = 400  # drop bulk rows with an implausibly long gap since the prior wave

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

    # repeat each diff once per simultaneous event at that timestamp
    intervals = np.repeat(diffs, counts[1:])

    return intervals


def get_transition_duration_table_edge(events_df, eventtype, group_col="clutch_id", date_col="night_date"):
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
    if eventtype not in EVENTTYPES:
        raise ValueError("eventtype must be 'sleep' or 'wake'")

    sub = events_df[events_df["event_type"] == eventtype]
    sub = sub.dropna(subset=["event_time", date_col, group_col]).copy()
    if sub.empty:
        return pd.DataFrame()

    cohort = [date_col, group_col]
    rows = []

    for _, subdf in sub.groupby(cohort):
        subdf_event = subdf.sort_values("event_time")

        times = subdf_event["event_time"].to_numpy()
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

            transitioners = subdf_event[subdf_event["event_time"] == this_time]

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


def get_transition_duration_tables_bulk(events_df, group_col="clutch_id", date_col="night_date"):
    """
    Bulk (in-night) events, both eventtypes in one per-animal state pass.
    Animals transition repeatedly, so the at-risk pool for `eventtype`
    (n_risk, aka M_r) is the live count of tracked animals in the
    complementary state, not a monotonically shrinking pool. Each animal
    starts being tracked at its own first event that night, seeded to that
    event's complement state - entry and exit are both just "the count":
    an animal simply stops contributing once its own bulk stream (already
    restricted to BULK_EXCLUSION_WINDOW_MIN minutes from its own edges,
    upstream in analyses.split_edge_bulk_events) runs out, no separate
    censoring needed.

    Same-timestamp events are batched, so interval_dur is always a gap
    between distinct times. Sleep/wake edge exclusion is independent per
    type, so a retained event can repeat an animal's known state (the
    opposite-type flip between them got excluded); such repeats aren't real
    transitions and are dropped.

    Returns:
        {"sleep": table, "wake": table}. n_risk is stored under the column
        name n_left, for parity with get_transition_duration_table_edge.
        Rows with interval_dur > BULK_MAX_INTERVAL_DUR_SEC are dropped.
    """
    df = events_df.dropna(subset=["event_time", date_col, group_col, "animal_id", "event_type"]).copy()
    if df.empty:
        return {eventtype: pd.DataFrame() for eventtype in EVENTTYPES}

    cohort = [date_col, group_col]
    df = df.sort_values(cohort + ["event_time"])

    rows = {eventtype: [] for eventtype in EVENTTYPES}
    for _, cohort_df in tqdm(df.groupby(cohort, sort=False), desc="bulk duration tables"):
        records = cohort_df.to_dict("records")

        state = {}
        occupancy = {"sleep": 0, "wake": 0}
        prev_time = {"sleep": None, "wake": None}

        i, n = 0, len(records)
        while i < n:
            event_time = records[i]["event_time"]
            j = i
            while j < n and records[j]["event_time"] == event_time:
                j += 1
            batch = records[i:j]
            i = j

            for rec in batch:
                animal, etype = rec["animal_id"], rec["event_type"]
                if animal not in state:
                    prior = _COMPLEMENT[etype]
                    state[animal] = prior
                    occupancy[prior] += 1

            genuine = [rec for rec in batch if state[rec["animal_id"]] != rec["event_type"]]

            n_total = len(state)
            for eventtype in EVENTTYPES:
                transitioners = [rec for rec in genuine if rec["event_type"] == eventtype]
                if not transitioners:
                    continue
                if prev_time[eventtype] is not None:
                    n_risk = occupancy[_COMPLEMENT[eventtype]]

                    if len(transitioners) > n_risk:
                        raise RuntimeError(
                            f"More transitions than animals occupying that state before the wave "
                            f"({eventtype}, {n_risk=}, transitioners={len(transitioners)})"
                        )

                    proportion_transitioned = len(transitioners) / n_risk
                    interval_dur = (event_time - prev_time[eventtype]) / np.timedelta64(1, "s")
                    for rec in transitioners:
                        row_out = dict(rec)
                        row_out.update({
                            "n_total": n_total,
                            "n_left": n_risk,
                            "proportion_transitioned": proportion_transitioned,
                            "interval_dur": interval_dur,
                            "eventtype": eventtype,
                        })
                        rows[eventtype].append(row_out)
                prev_time[eventtype] = event_time

            for rec in genuine:
                animal, etype = rec["animal_id"], rec["event_type"]
                occupancy[state[animal]] -= 1
                state[animal] = etype
                occupancy[etype] += 1

    tables = {eventtype: pd.DataFrame(rows[eventtype]) for eventtype in EVENTTYPES}
    print(tables)
    quit()
    return {eventtype: table[table["interval_dur"] <= BULK_MAX_INTERVAL_DUR_SEC].reset_index(drop=True)
            if not table.empty else table
            for eventtype, table in tables.items()}


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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    import analyses
    import preprocessing

    FOCAL_GROUP = None              # None = first clutch_id in the data
    FOCAL_DATE = None               # None = that clutch's earliest night_date

    masterdf = preprocessing.load_regular_data()
    FOCAL_GROUP = FOCAL_GROUP or masterdf["clutch_id"].iloc[0]
    masterdf = masterdf[masterdf["clutch_id"] == FOCAL_GROUP]

    edge_events = analyses.build_edge_events_from_masterdf(masterdf)
    bulk_events = analyses.build_bulk_events(masterdf, edge_events)

    edge_tables_all = {et: get_transition_duration_table_edge(edge_events, et) for et in EVENTTYPES}
    bulk_tables_all = get_transition_duration_tables_bulk(bulk_events)

    FOCAL_DATE = FOCAL_DATE or sorted(masterdf["night_date"].unique())[0]
    tables = {
        "edge": {et: t[t["night_date"] == FOCAL_DATE] for et, t in edge_tables_all.items()},
        "bulk": {et: t[t["night_date"] == FOCAL_DATE] for et, t in bulk_tables_all.items()},
    }
    colors = {"edge_sleep": "#1b9e77", "edge_wake": "#d95f02",
              "bulk_sleep": "#7570b3", "bulk_wake": "#e7298a"}

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, model_tables in tables.items():
        for eventtype, table in model_tables.items():
            if table.empty:
                continue
            sub = table.sort_values("event_time")
            ax.step(sub["event_time"], sub["n_left"], where="post", marker="o", markersize=4,
                    label=f"{model}_{eventtype}", color=colors[f"{model}_{eventtype}"])
            print(f"--- {model} {eventtype} ---")
            print(sub[["event_time", "n_total", "n_left", "proportion_transitioned", "interval_dur"]]
                  .to_string(index=False))

    ax.set_xlabel("event_time")
    ax.set_ylabel("n_left (at-risk pool)")
    ax.set_title(f"{FOCAL_GROUP}, {FOCAL_DATE}: edge vs bulk at-risk pool")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()
