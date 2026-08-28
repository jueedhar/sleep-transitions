# Juee Dhar
# 28 Aug 2026


import numpy as np
import pandas as pd


def _derange(dates, rng):
    """Shuffle so no date stays where it was, by swapping any that did."""
    n = len(dates)
    if n < 2:
        return dates.copy()
    order = rng.permutation(n)
    for i in np.flatnonzero(order == np.arange(n)):
        j = rng.integers(0, n - 1)
        j = j + 1 if j >= i else j
        order[i], order[j] = order[j], order[i]
    return dates[order]


def make_date_map(events_df, seed=None, derange=True):
    """
    For each animal, shuffle its own night_dates among themselves -- a bijection,
    so no date is repeated, lost or invented. derange=True forbids a night keeping its own date.
    
    Returns (animal_id, night_date, new_night_date).
    """
    rng = np.random.default_rng(seed)
    pairs = events_df[["animal_id", "night_date"]].drop_duplicates()

    frames = []
    for animal_id, sub in pairs.groupby("animal_id"):
        dates = np.sort(sub["night_date"].to_numpy())
        new = _derange(dates, rng) if derange else rng.permutation(dates)
        frames.append(pd.DataFrame({"animal_id": animal_id,
                                    "night_date": dates,
                                    "new_night_date": new}))

    return pd.concat(frames, ignore_index=True)


def apply_date_map(events_df, date_map):
    """
    event_time is shifted by the same number of days, so the time of night is kept exactly and only the 
    date changes.
    """
    df = events_df.merge(date_map, on=["animal_id", "night_date"], how="left")
    df["new_night_date"] = df["new_night_date"].fillna(df["night_date"])

    df["event_time"] = df["event_time"] + (df["new_night_date"] - df["night_date"])
    df["night_date"] = df["new_night_date"]
    return df.drop(columns=["new_night_date"]).reset_index(drop=True)