#03 Sept 2026
#Juee Dhar

"""Memory in sleep/wake onset timing and order with clutch, via ACF and lagged cross-correlation."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

import config
import preprocessing

df = preprocessing.load_regular_data()

FIGURES_DIR = os.path.join(config.FIGURES, "acf")
os.makedirs(FIGURES_DIR, exist_ok=True)

IND = "animal_id"
CLUTCH = "clutch_id"
DATE = "night_date"
ONSET = "t_sleep"
WAKING = "t_wake"

NLAGS = 10
MIN_NIGHTS = 40       # min nights of data before trusting a correlation
MIN_CLUTCH_DATES = 3  # min distinct nights before trusting a clutch's rank ACF

df = df.sort_values(DATE)
assert pd.api.types.is_datetime64_any_dtype(df[DATE]), f"{DATE} is not parsed as datetime"
assert df.groupby(IND)[DATE].apply(lambda s: s.is_monotonic_increasing).all(), \
    f"{DATE} is not monotonically increasing within at least one {IND} after sort_values"
print(f"{DATE} verified as datetime64 and non-decreasing within every {IND}")


def to_hours_since_noon(ts_col):
    hour = ts_col.dt.hour + ts_col.dt.minute / 60
    return np.where(hour < 12, hour + 24, hour)


ONSET_H, WAKE_H = "onset_hours", "wake_hours"
df[ONSET_H] = to_hours_since_noon(df[ONSET])
df[WAKE_H] = to_hours_since_noon(df[WAKING])


# core functions

def bartlett_ci(n):
    return 1.96 / (n ** 0.5)


def acf_per_series(df, value_col, id_col, date_col, nlags, min_nights=MIN_NIGHTS):
    """One ACF curve per id_col group. Long format: one row per (group, lag)."""
    rows = []
    for group_id, sub in df.groupby(id_col):
        series = sub.sort_values(date_col)[value_col].dropna()
        n = len(series)
        if n < min_nights:
            continue
        values = acf(series.to_numpy(), nlags=nlags, fft=False)
        ci = bartlett_ci(n)
        for lag, value in enumerate(values):
            if lag == 0:
                continue
            rows.append({id_col: group_id, "lag": lag, "acf": value, "n_nights": n,
                         "ci_95": ci, "significant": abs(value) > ci})
    return pd.DataFrame(rows)


def lagged_corr_per_animal(df, col_a, col_b, id_col, date_col, lag, min_nights=MIN_NIGHTS):
    """corr(col_a(t), col_b(t+lag)) per animal. Same row shape as acf_per_series."""
    rows = []
    for animal_id, sub in df.groupby(id_col):
        sub = sub.sort_values(date_col)
        paired = pd.DataFrame({"a": sub[col_a].to_numpy(),
                                "b": sub[col_b].shift(-lag).to_numpy()}).dropna()
        n = len(paired)
        if n < min_nights:
            continue
        r = paired["a"].corr(paired["b"])
        ci = bartlett_ci(n)
        rows.append({id_col: animal_id, "lag": lag, "corr": r, "n_nights": n,
                     "ci_95": ci, "significant": abs(r) > ci})
    return pd.DataFrame(rows)


# plotting 

def distinct_colors(n):
    """n visually distinct colors, how many ever groups there are (no repeats)."""
    if n <= 20:
        return plt.colormaps["tab20"].colors[:n]
    return plt.colormaps["gist_ncar"](np.linspace(0, 0.95, n))


def lag_summary(sub, value_col):
    """Mean +/- SEM per lag. sem is NaN for a single-member group (n=1) --
    zero it out, since a NaN errorbar breaks savefig's tight-bbox layout."""
    summary = sub.groupby("lag")[value_col].agg(["mean", "sem"]).reset_index()
    summary["sem"] = summary["sem"].fillna(0)
    return summary


def plot_acf_on_ax(ax, acf_df, id_col, title, value_col="acf", group_by=None):
    for _, sub in acf_df.groupby(id_col):
        sub = sub.sort_values("lag")
        ax.plot(sub["lag"], sub[value_col], color="gray", alpha=0.2, linewidth=0.8)

    if group_by is None:
        summary = lag_summary(acf_df, value_col)
        ax.errorbar(summary["lag"], summary["mean"], yerr=summary["sem"],
                     color="black", marker="o", linewidth=1.5, capsize=2, label="mean")
    else:
        groups = list(acf_df.groupby(group_by))
        for color, (group_val, sub) in zip(distinct_colors(len(groups)), groups):
            summary = lag_summary(sub, value_col)
            ax.errorbar(summary["lag"], summary["mean"], yerr=summary["sem"], color=color,
                         marker="o", linewidth=1.5, capsize=2, label=str(group_val))
        ax.legend(fontsize=7, frameon=False, title=group_by)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    if "ci_95" in acf_df.columns:
        # per-animal CI varies with that animal's n; this is the mean threshold
        # across animals at each lag -- see the "significant" column for exact.
        mean_ci = acf_df.groupby("lag")["ci_95"].mean()
        ax.plot(mean_ci.index, mean_ci.values, color="red", linestyle=":", linewidth=1, label="mean 95% CI")
        ax.plot(mean_ci.index, -mean_ci.values, color="red", linestyle=":", linewidth=1)
        ax.legend(fontsize=7, frameon=False)

    ax.set_xlabel("lag (nights)")
    ax.set_ylabel(value_col)
    ax.set_title(title, fontsize=9)
    return ax


def plot_mean_acf(acf_df, id_col, title, fname, value_col="acf", group_by=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_acf_on_ax(ax, acf_df, id_col, title, value_col=value_col, group_by=group_by)
    fig.tight_layout()

    for ext in (".png", ".svg"):
        fig.savefig(os.path.join(FIGURES_DIR, fname + ext), dpi=150 if ext == ".png" else None,
                     bbox_inches="tight")
    parquet_path = os.path.join(FIGURES_DIR, fname + ".parquet")
    acf_df.to_parquet(parquet_path)
    print("saved", fname, "(.png/.svg) and", parquet_path)
    return fig, ax


# ACF: onset vs itself, wake vs itself
for label, col in (("onset", ONSET_H), ("wake", WAKE_H)):
    acf_by_animal = acf_per_series(df, value_col=col, id_col=IND, date_col=DATE, nlags=NLAGS)
    plot_mean_acf(acf_by_animal, id_col=IND,
                  title=f"ACF of {label} time across nights, per animal",
                  fname=f"acf_{label}_by_animal")


#  CCF: does wake(t) afect onset(t) and onset(t) affect wake(t+1)

for (a_label, a_col), (b_label, b_col) in [
    (("wake", WAKE_H), ("onset", ONSET_H)),
    (("onset", ONSET_H), ("wake", WAKE_H)),
]:
    corr_df = pd.concat(
        [lagged_corr_per_animal(df, col_a=a_col, col_b=b_col, id_col=IND, date_col=DATE, lag=lag)
         for lag in range(NLAGS + 1)],
        ignore_index=True,
    )
    plot_mean_acf(corr_df, id_col=IND,
                  title=f"corr({a_label}(t), {b_label}(t+lag)) per animal",
                  fname=f"lagged_corr_{a_label}_to_{b_label}", value_col="corr")


# ACF of within-clutch order 
# Rank is per (clutch, night) among whoever is present that night - if night 1 has ABCD and night 2 has BC, B and C still get 
# a valid relative rank both nights. Only requirement: enough nights per clutch to ask whether that order holds over time.

n_dates_per_clutch = df.groupby(CLUTCH)[DATE].nunique()
keep_clutches = n_dates_per_clutch[n_dates_per_clutch >= MIN_CLUTCH_DATES].index
for clutch_id, n_dates in n_dates_per_clutch[n_dates_per_clutch < MIN_CLUTCH_DATES].items():
    print(f"dropping clutch {clutch_id!r}: only {n_dates} date(s)")
print(f"kept {len(keep_clutches)} / {len(n_dates_per_clutch)} clutches")

clean_df = df[df[CLUTCH].isin(keep_clutches)].copy()

for label, col in (("onset", ONSET_H), ("wake", WAKE_H)):
    rank_col = f"{label}_order_in_clutch"
    clean_df[rank_col] = clean_df.groupby([CLUTCH, DATE])[col].rank(method="average")

    rank_check = clean_df.groupby([CLUTCH, DATE])[rank_col].agg(["min", "max", "count"])
    assert (rank_check["min"] >= 1).all() and (rank_check["max"] <= rank_check["count"]).all(), \
        f"{rank_col} leaked outside its (clutch_id, night_date) group"
    print(f"{rank_col} verified within {len(rank_check)} (clutch_id, night_date) groups")

    acf_order_by_animal = acf_per_series(clean_df, value_col=rank_col, id_col=IND,
                                          date_col=DATE, nlags=NLAGS)
    animal_to_clutch = clean_df[[IND, CLUTCH]].drop_duplicates(subset=IND)
    acf_order_by_animal = acf_order_by_animal.merge(animal_to_clutch, on=IND, how="left")

    plot_mean_acf(acf_order_by_animal, id_col=IND,
                  title=f"ACF of within-clutch {label} rank across nights, by clutch",
                  fname=f"acf_{label}_order_in_clutch_by_animal", group_by=CLUTCH)
