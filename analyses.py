# Juee Dhar 07 Sept 2026
# Pranav Minasandra March 23, 2026

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import config
import durations
import estimation


BULK_EXCLUSION_WINDOW_MIN = 30
LOCAL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOCAL_TZ = "Africa/Nairobi"  # fixed UTC+3, no DST

EVENT_META_COLS = ["animal_id", "night_date", "clutch_id", "group_id", "size_class", "coverage_class", "sleep_site_type", "wake_site_type", "age", "sex"]
EVENTTYPES = durations.EVENTTYPES


# Events split into edge and bulk

# One row per animal-night edge event (sleep, wake), long format with metadata.
def build_edge_events_from_masterdf(masterdf):
    other_meta = [c for c in EVENT_META_COLS if c not in ("animal_id", "night_date")]
    night_table = masterdf[["animal_id", "night_date", "t_sleep", "t_wake"] + other_meta].copy()
    night_table["age_sex"] = np.where(night_table["age"].isna() | night_table["sex"].isna(), np.nan,
                                      night_table["age"].astype(str) + "_" + night_table["sex"].astype(str))

    meta_cols = EVENT_META_COLS + ["age_sex"]
    sleep_rows = night_table.rename(columns={"t_sleep": "event_time"}).assign(event_type="sleep")
    wake_rows = night_table.rename(columns={"t_wake": "event_time"}).assign(event_type="wake")

    events = pd.concat([sleep_rows[["event_time", "event_type"] + meta_cols],
                        wake_rows[["event_time", "event_type"] + meta_cols]], ignore_index=True)
    return events.dropna(subset=["event_time"]).reset_index(drop=True)


# Parses local_time strings to datetime, falling back to mixed format on mismatches.
def _parse_local_time(series):
    parsed = pd.to_datetime(series, format=LOCAL_TIME_FORMAT, errors="coerce")
    bad = parsed.isna() & series.notna()
    if bad.any():
        parsed.loc[bad] = pd.to_datetime(series.loc[bad], format="mixed")
    return parsed


# One row per state flip (sleep_bouts change) for one animal, across all its nights.
def _extract_flips_for_individual(df, animal_id):
    df = df.copy()
    # raw parquet's `local_time` is a bare time-of-day string with no date; parsing it via
    # _parse_local_time silently defaulted the missing date to today's date, so every bulk
    # event_time landed on whatever day the script ran rather than its real date (debugged
    # 23 Sept 2026). `timestamp` is tz-aware UTC and carries the correct date, so derive local
    # time from that instead. This feeds get_transition_duration_tables_bulk in durations.py,
    # via build_bulk_events -> split_edge_bulk_events below.
    df["local_time"] = df["timestamp"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    df = df.sort_values("local_time").reset_index(drop=True)

    state = df["sleep_bouts"].to_numpy()
    times = df["local_time"].to_numpy()
    night_dates = pd.to_datetime(df["night_date"]).to_numpy()

    valid = ~pd.isna(state)
    state, times, night_dates = state[valid], times[valid], night_dates[valid]
    change_idx = np.where(np.diff(state) != 0)[0] + 1

    return pd.DataFrame({
        "animal_id": animal_id,
        "event_time": times[change_idx],
        "night_date": night_dates[change_idx],
        "event_type": np.where(state[change_idx] == 1, "sleep", "wake"),
    })


# Keeps only flip events >= exclusion_window_min from their same-type edge event, and further
# restricts to the clutch-night's core-sleep window: after the last individual's sleep onset and
# before the first individual's wake, i.e. the span where every tracked animal is asleep. Outside
# that window a "bulk" event is daytime/pre-sleep/post-wake leakage, not a real in-night
# transition (this used to be unenforced -- diagnose_bulk_window.py measured ~24% leakage on
# clutch 18 before this was added).
def split_edge_bulk_events(full_events, edge_events, group_col="clutch_id", date_col="night_date",
                           exclusion_window_min=BULK_EXCLUSION_WINDOW_MIN):
    if full_events.empty:
        return full_events.copy()

    merged = full_events.merge(
        edge_events[["animal_id", "night_date", "event_type", "event_time"]]
            .rename(columns={"event_time": "edge_time"}),
        on=["animal_id", "night_date", "event_type"], how="left")

    edge_keys = edge_events[["animal_id", "night_date"]].drop_duplicates().assign(has_edge=True)
    merged = merged.merge(edge_keys, on=["animal_id", "night_date"], how="left")
    merged["has_edge"] = merged["has_edge"].notna()

    minutes_from_edge = (merged["event_time"] - merged["edge_time"]).abs() / pd.Timedelta(minutes=1)
    near_edge = minutes_from_edge.le(exclusion_window_min).fillna(False)

    bulk = merged[~near_edge & merged["has_edge"]].drop(columns=["edge_time", "has_edge"])

    cohort = [date_col, group_col]
    onsets = edge_events[edge_events["event_type"] == "sleep"]
    offsets = edge_events[edge_events["event_type"] == "wake"]
    window = onsets.groupby(cohort)["event_time"].max().rename("t_start").reset_index()
    window = window.merge(offsets.groupby(cohort)["event_time"].min().rename("t_end").reset_index(), on=cohort)
    window = window[window["t_end"] > window["t_start"]]

    bulk = bulk.merge(window, on=cohort, how="inner")
    bulk = bulk[(bulk["event_time"] >= bulk["t_start"]) & (bulk["event_time"] < bulk["t_end"])]
    return bulk.drop(columns=["t_start", "t_end"]).reset_index(drop=True)


# Reads each animal's inactivity parquet, extracts flips, and returns the bulk 
def build_bulk_events(masterdf, edge_events, inactivity_dir=None,
                      exclusion_window_min=BULK_EXCLUSION_WINDOW_MIN):
    if inactivity_dir is None:
        inactivity_dir = os.path.join(config.DATA, "inactivity")

    night_table = masterdf[EVENT_META_COLS].drop_duplicates(["animal_id", "night_date"]).copy()
    night_table["age_sex"] = np.where(night_table["age"].isna() | night_table["sex"].isna(), np.nan,
                                      night_table["age"].astype(str) + "_" + night_table["sex"].astype(str))

    per_animal = []
    for animal_id in tqdm(night_table["animal_id"].unique(), desc="animals (bulk)"):
        path = os.path.join(inactivity_dir, f"{animal_id}.parquet")
        if not os.path.exists(path):
            print(f"Missing parquet for {animal_id}, skipped")
            continue
        try:
            per_animal.append(_extract_flips_for_individual(pd.read_parquet(path), animal_id))
        except KeyError as e:
            print(f"Skipping {animal_id}: missing column {e} in {path}")

    if not per_animal:
        return pd.DataFrame()

    full_events = pd.concat(per_animal, ignore_index=True).merge(
        night_table, on=["animal_id", "night_date"], how="inner")

    return split_edge_bulk_events(full_events, edge_events, exclusion_window_min=exclusion_window_min)


# Labels each event early/mid/late by its fractional position within that clutch-night's span.
def assign_night_third(events_df, time_col="event_time", date_col="night_date", clutch_col="clutch_id"):
    df = events_df.copy()
    bounds = df.groupby([date_col, clutch_col])[time_col].agg(["min", "max"])
    df = df.merge(bounds, on=[date_col, clutch_col], how="left")

    span = (df["max"] - df["min"]) / np.timedelta64(1, "s")
    elapsed = (df[time_col] - df["min"]) / np.timedelta64(1, "s")
    frac = (elapsed / span.replace(0, np.nan)).clip(0, 1).fillna(0)

    df["night_third"] = pd.cut(frac, bins=[-0.001, 1 / 3, 2 / 3, 1.0],
                               labels=["early", "mid", "late"]).astype(str)
    return df.drop(columns=["min", "max"])


# Estimation

EST_COLS = ["label", "eventtype", "percentile_bin", "p_estimate", "p_error",
            "n_individuals", "n_clutch_nights"]


def build_duration_tables(events_df, kind="edge", group_col="clutch_id", date_col="night_date"):
    """
    {eventtype: duration table}, built once from the whole cohort.
    `kind` selects the at-risk model: "edge" (monotonic pool) or "bulk"
    (per-animal state tracking, both eventtypes in one simulation pass).
    """
    if kind == "bulk":
        tables = durations.get_transition_duration_tables_bulk(
            events_df, group_col=group_col, date_col=date_col)
    else:
        tables = {eventtype: durations.get_transition_duration_table_edge(
                       events_df, eventtype, group_col=group_col, date_col=date_col)
                   for eventtype in EVENTTYPES}
    return {eventtype: table for eventtype, table in tables.items() if not table.empty}


def compute_estimates(tables, by="none", date_col="night_date", group_col="clutch_id",
                      drop_vals=("Unknown",), percentile_bins=config.PERCENTILE_THRESHOLDS,
                      n_boot=20):
    """
    Rate estimates (p_estimate, p_error) per eventtype x percentile_bin, one
    row per label. `by` only decides which rows' durations feed each rate
    estimate -- the cohort (n_left, percentile_bin) is untouched.

    n_clutch_nights counts distinct (date_col, group_col) pairs surviving in
    `t` -- the unique clutch-nights, not just distinct dates (different
    clutches on the same calendar date are different cohorts).
    """
    frames = []
    for eventtype, table in tables.items():
        t = table
        if by != "none":
            if by not in t.columns:
                raise ValueError(f"'{by}' is not a column in the events table")
            t = t[t[by].notna() & ~t[by].isin(drop_vals)]
            if t.empty:
                continue

        est = estimation.estimate_exp_by_percentile_df(
            df=t, percentile_bins=percentile_bins, n_boot=n_boot, foreach=by)
        if est.empty:
            continue

        est = est.copy()
        est["eventtype"] = eventtype
        est["label"] = "all" if by == "none" else est[by].astype(str)

        if by == "none":
            est["n_individuals"] = t["animal_id"].nunique()
            est["n_clutch_nights"] = t[[date_col, group_col]].drop_duplicates().shape[0]
        else:
            cohort = list(zip(t[date_col], t[group_col]))
            counts = (t.assign(_cohort=cohort).groupby(by)
                       .agg(n_individuals=("animal_id", "nunique"),
                           n_clutch_nights=("_cohort", "nunique"))
                       .reset_index())
            counts["label"] = counts[by].astype(str)
            est = est.drop(columns=[by]).merge(
                counts[["label", "n_individuals", "n_clutch_nights"]], on="label", how="left")
        frames.append(est)

    if not frames:
        return pd.DataFrame(columns=EST_COLS)
    return pd.concat(frames, ignore_index=True)[EST_COLS]


# Plotting

def _counts_text(est, label):
    """The plot headings show no. of individuals and no. of clutch-nights."""
    row = est[est["label"] == label]
    if row.empty:
        return ""
    return (f"n={int(row['n_individuals'].iloc[0])} individuals, "
           f"{int(row['n_clutch_nights'].iloc[0])} clutch-nights")


def _line(ax, sub, name, color, linestyle, alpha, y_scale="p"):
    """Draws one p_estimate-vs-percentile_bin line with error bars onto `ax`."""
    sub = sub.sort_values("percentile_bin")
    y, yerr, ylabel = sub["p_estimate"], sub["p_error"], "p_estimate"
    if y_scale == "logit":
        yerr = yerr / (y * (1 - y))
        y = np.log(y / (1 - y))
        ylabel = "logit(p_estimate)"
    ax.plot(sub["percentile_bin"], y, marker="o", linewidth=0.7,
            linestyle=linestyle, alpha=alpha, label=name, color=color)
    ax.errorbar(sub["percentile_bin"], y, yerr=yerr,
                fmt="none", capsize=2, linewidth=0.6, color=color, alpha=alpha)
    ax.set_xlabel("percentile_bin")
    ax.set_ylabel(ylabel)


def plot_eventtype_panels(est, axes=None, linestyle="-", alpha=1.0, suffix="", set_titles=True,
                          y_scale="p"):
    """Two panels (sleep und wake); one line per label within each."""
    sns.set_theme(style="whitegrid")
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    labels = sorted(est["label"].unique())
    colors = dict(zip(labels, sns.color_palette(n_colors=max(len(labels), 1))))

    for ax, eventtype in zip(axes, EVENTTYPES):
        sub = est[est["eventtype"] == eventtype]
        for label in labels:
            line_df = sub[sub["label"] == label]
            if not line_df.empty:
                _line(ax, line_df, f"{label}{suffix}", colors[label], linestyle, alpha, y_scale)
        if set_titles:
            parts = [f"{lab}: {_counts_text(sub, lab)}" for lab in labels if not sub[sub["label"] == lab].empty]
            ax.set_title(f"{eventtype}\n" + " | ".join(parts), fontsize=8)
        ax.legend(fontsize=7, frameon=False)

    if fig is not None:
        fig.tight_layout()
    return fig, axes


def plot_category_panels(est, axes=None, linestyle="-", alpha=1.0, suffix="", set_titles=True,
                         y_scale="p"):
    """One panel per label; sleep and wake as two separate lines within each."""
    sns.set_theme(style="whitegrid")
    labels = sorted(est["label"].unique())

    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, max(len(labels), 1), figsize=(6 * max(len(labels), 1), 4),
                                 sharey=True, squeeze=False)
        axes = axes[0]

    colors = dict(zip(EVENTTYPES, sns.color_palette(n_colors=len(EVENTTYPES))))

    for ax, label in zip(axes, labels):
        sub = est[est["label"] == label]
        for eventtype in EVENTTYPES:
            line_df = sub[sub["eventtype"] == eventtype]
            if not line_df.empty:
                _line(ax, line_df, f"{eventtype}{suffix}", colors[eventtype], linestyle, alpha, y_scale)
        if set_titles:
            ax.set_title(f"{label}\n{_counts_text(sub, label)}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)

    if fig is not None:
        fig.tight_layout()
    return fig, axes


def plot_bulk_interval_duration(tables, group_col="clutch_id", date_col="night_date",
                                max_interval_dur=durations.BULK_MAX_INTERVAL_DUR_SEC,
                                bin_centers=range(100, 701, 100), bin_halfwidth=50):
    """
    interval_dur distribution (violin) across discretized time-of-night bins -- each bin
    is bin_center +- bin_halfwidth minutes since that clutch-night's own first bulk event
    (default: 100, 200, ... 700 +- 50 min). One panel per eventtype (sleep, wake).
    `max_interval_dur` re-gates on top of durations.BULK_MAX_INTERVAL_DUR_SEC (already
    applied when `tables` was built), so callers can tighten the cutoff for a plot without
    touching durations.py. Takes already-built tables -- does not recompute them, so it's
    cheap to call alongside other plots on the same tables.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    colors = dict(zip(EVENTTYPES, sns.color_palette(n_colors=len(EVENTTYPES))))
    cohort = [date_col, group_col]

    bin_centers = list(bin_centers)
    edges = [c - bin_halfwidth for c in bin_centers] + [bin_centers[-1] + bin_halfwidth]

    for ax, eventtype in zip(axes, EVENTTYPES):
        table = tables.get(eventtype, pd.DataFrame())
        if table.empty:
            ax.set_title(f"{eventtype}\n(no data)")
            continue

        sub = table[table["interval_dur"] <= max_interval_dur].copy()
        night_start = sub.groupby(cohort)["event_time"].transform("min")
        sub["elapsed_min"] = (sub["event_time"] - night_start) / np.timedelta64(1, "m")
        n_clutch_nights = sub[cohort].drop_duplicates().shape[0]

        sub["time_bin"] = pd.cut(sub["elapsed_min"], bins=edges, labels=bin_centers)
        sub = sub.dropna(subset=["time_bin"])

        sns.violinplot(data=sub, x="time_bin", y="interval_dur", ax=ax,
                       color=colors[eventtype], cut=0)

        ax.set_title(f"{eventtype}\n{n_clutch_nights} clutch-nights", fontsize=9)
        ax.set_xlabel(f"minutes since that clutch-night's first bulk event "
                      f"(binned, +-{bin_halfwidth}min)")

    axes[0].set_ylabel("interval_dur (s)")
    fig.tight_layout()
    return fig, axes

