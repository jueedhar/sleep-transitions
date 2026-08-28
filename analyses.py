# Juee Dhar 25 Aug 2026
# Pranav Minasandra March 23, 2026

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import config
import estimation


BULK_EXCLUSION_WINDOW_MIN = 30
LOCAL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

EVENT_META_COLS = ["animal_id", "night_date", "clutch_id", "group_id", "size_class", "sleep_site_type", "wake_site_type", "age", "sex"]
EVENTTYPES = ("sleep", "wake")


# Events split into edge and bulk

def build_edge_events_from_masterdf(masterdf):
    other_meta = [c for c in EVENT_META_COLS if c not in ("animal_id", "night_date")]
    night_table = masterdf[["animal_id", "night_date", "t_sleep", "t_wake"] + other_meta].copy()
    night_table["age_sex"] = night_table["age"].astype(str) + "_" + night_table["sex"].astype(str)

    meta_cols = EVENT_META_COLS + ["age_sex"]
    sleep_rows = night_table.rename(columns={"t_sleep": "event_time"}).assign(event_type="sleep")
    wake_rows = night_table.rename(columns={"t_wake": "event_time"}).assign(event_type="wake")

    events = pd.concat([sleep_rows[["event_time", "event_type"] + meta_cols],
                        wake_rows[["event_time", "event_type"] + meta_cols]], ignore_index=True)
    return events.dropna(subset=["event_time"]).reset_index(drop=True)


def _parse_local_time(series):
    parsed = pd.to_datetime(series, format=LOCAL_TIME_FORMAT, errors="coerce")
    bad = parsed.isna() & series.notna()
    if bad.any():
        parsed.loc[bad] = pd.to_datetime(series.loc[bad], format="mixed")
    return parsed


def _extract_flips_for_individual(df, animal_id):
    df = df.copy()
    df["local_time"] = _parse_local_time(df["local_time"])
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


def split_edge_bulk_events(full_events, edge_events, exclusion_window_min=BULK_EXCLUSION_WINDOW_MIN):
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
    return bulk.reset_index(drop=True)


def build_bulk_events(masterdf, edge_events, inactivity_dir=None,
                      exclusion_window_min=BULK_EXCLUSION_WINDOW_MIN):
    if inactivity_dir is None:
        inactivity_dir = os.path.join(config.DATA, "inactivity")

    night_table = masterdf[EVENT_META_COLS].drop_duplicates(["animal_id", "night_date"]).copy()
    night_table["age_sex"] = night_table["age"].astype(str) + "_" + night_table["sex"].astype(str)

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


def assign_night_third(events_df, time_col="event_time", date_col="night_date"):
    df = events_df.copy()
    bounds = df.groupby(date_col)[time_col].agg(["min", "max"])
    df = df.merge(bounds, on=date_col, how="left")

    span = (df["max"] - df["min"]) / np.timedelta64(1, "s")
    elapsed = (df[time_col] - df["min"]) / np.timedelta64(1, "s")
    frac = (elapsed / span.replace(0, np.nan)).clip(0, 1).fillna(0)

    df["night_third"] = pd.cut(frac, bins=[-0.001, 1 / 3, 2 / 3, 1.0],
                               labels=["early", "mid", "late"]).astype(str)
    return df.drop(columns=["min", "max"])


# Estimation

def get_transition_duration_table(events_df, eventtype, group_col="group_id", date_col="night_date"):
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


EST_COLS = ["label", "eventtype", "percentile_bin", "p_estimate", "p_error",
            "n_individuals", "n_nights"]


def build_duration_tables(events_df, group_col="group_id", date_col="night_date"):
    """
    {eventtype: duration table}, built once from the whole cohort.
    """
    tables = {}
    for eventtype in EVENTTYPES:
        table = get_transition_duration_table(events_df, eventtype,
                                              group_col=group_col, date_col=date_col)
        if not table.empty:
            tables[eventtype] = table
    return tables


def compute_estimates(tables, by="none", date_col="night_date", drop_vals=("Unknown",),
                      percentile_bins=config.PERCENTILE_THRESHOLDS, n_boot=20):
    """
    `by` only decides which rows' durations feed each rate estimate -- the
    cohort (n_left, percentile_bin) is untouched.
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
            est["n_nights"] = t[date_col].nunique()
        else:
            counts = t.groupby(by).agg(n_individuals=("animal_id", "nunique"),
                                       n_nights=(date_col, "nunique")).reset_index()
            counts["label"] = counts[by].astype(str)
            est = est.drop(columns=[by]).merge(counts[["label", "n_individuals", "n_nights"]],
                                               on="label", how="left")
        frames.append(est)

    if not frames:
        return pd.DataFrame(columns=EST_COLS)
    return pd.concat(frames, ignore_index=True)[EST_COLS]


# Plotting

def _counts_text(est, label):
    """ The plot headings show no. of individuals and no. of nights"""
    row = est[est["label"] == label]
    if row.empty:
        return ""
    return f"n={int(row['n_individuals'].iloc[0])} individuals, {int(row['n_nights'].iloc[0])} nights"


def _line(ax, sub, name, color, linestyle, alpha):
    sub = sub.sort_values("percentile_bin")
    ax.plot(sub["percentile_bin"], sub["p_estimate"], marker="o", linewidth=0.7,
            linestyle=linestyle, alpha=alpha, label=name, color=color)
    ax.errorbar(sub["percentile_bin"], sub["p_estimate"], yerr=sub["p_error"],
                fmt="none", capsize=2, linewidth=0.6, color=color, alpha=alpha)
    ax.set_xlabel("percentile_bin")
    ax.set_ylabel("p_estimate")


def plot_eventtype_panels(est, axes=None, linestyle="-", alpha=1.0, suffix="", set_titles=True):
    """Two panels (sleep | wake); one line per label within each."""
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
                _line(ax, line_df, f"{label}{suffix}", colors[label], linestyle, alpha)
        if set_titles:
            parts = [f"{lab}: {_counts_text(sub, lab)}" for lab in labels if not sub[sub["label"] == lab].empty]
            ax.set_title(f"{eventtype}\n" + " | ".join(parts), fontsize=8)
        ax.legend(fontsize=7, frameon=False)

    if fig is not None:
        fig.tight_layout()
    return fig, axes


def plot_category_panels(est, axes=None, linestyle="-", alpha=1.0, suffix="", set_titles=True):
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
                _line(ax, line_df, f"{eventtype}{suffix}", colors[eventtype], linestyle, alpha)
        if set_titles:
            ax.set_title(f"{label}\n{_counts_text(sub, label)}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)

    if fig is not None:
        fig.tight_layout()
    return fig, axes

