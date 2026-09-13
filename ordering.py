#13 Sept 2026

"""Within-clutch order thirds (sleep onset, wake, TST) vs age-sex/sex class: likelihood and significance."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, friedmanchisquare
import analyses
import config
import preprocessing

FIGURES_DIR = os.path.join(config.FIGURES, "ordering")
os.makedirs(FIGURES_DIR, exist_ok=True)

CLUTCH = "clutch_id"
DATE = "night_date"
CLASS_COLS = ["age_sex", "sex"]

THIRDS = ["early", "mid", "late"]
TST_THIRDS = ["low", "mid", "high"]  # ascending TST: low = shortest TST third
THIRD_BINS = [-0.001, 1 / 3, 2 / 3, 1.0]


def assign_order_third(events, value_col, thirds, ascending=True, clutch_col=CLUTCH, date_col=DATE):
    """Ranks each animal's value within its (clutch, night), then bins that rank
    fraction into thirds -- same bin logic as analyses.assign_night_third, applied
    to order instead of clock time (ascending=True: lowest value ranked first)."""
    events = events.copy()
    rank = events.groupby([clutch_col, date_col])[value_col].rank(method="average", ascending=ascending)
    group_size = events.groupby([clutch_col, date_col])[value_col].transform("size")
    frac = rank / group_size
    events["order_third"] = pd.cut(frac, bins=THIRD_BINS, labels=thirds).astype(str)
    return events


def class_likelihoods(events, class_col, thirds, clutch_col=CLUTCH, date_col=DATE):
    """Per (clutch, night, class): fraction of that class's members falling in each third."""
    counts = (events.groupby([clutch_col, date_col, class_col, "order_third"])
                     .size().unstack(fill_value=0).reindex(columns=thirds, fill_value=0))
    likelihood = counts.div(counts.sum(axis=1), axis=0)
    return likelihood.reset_index().melt(id_vars=[clutch_col, date_col, class_col],
                                          var_name="order_third", value_name="likelihood")


# same as analyses.plot_eventtype_panels's per-label colors, so classes match the colors already used for the main edge/bulk figures
def class_colors(labels):
    sns.set_theme(style="whitegrid")
    labels = sorted(labels)
    return dict(zip(labels, sns.color_palette(n_colors=len(labels))))


def test_between_classes(likelihood_df, class_col, thirds):
    """Per third: are the classes different from each other (Kruskal-Wallis)?"""
    rows = []
    for third in thirds:
        groups = [g["likelihood"].to_numpy()
                  for _, g in likelihood_df[likelihood_df["order_third"] == third].groupby(class_col)]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        rows.append({"order_third": third, "statistic": stat, "p_value": p, "n_classes": len(groups)})
    return pd.DataFrame(rows)


def test_across_thirds(likelihood_df, class_col, thirds):
    """Per class: does likelihood differ across the 3 thirds (Friedman, paired by clutch-night)?"""
    rows = []
    for cls, sub in likelihood_df.groupby(class_col):
        wide = sub.pivot_table(index=[CLUTCH, DATE], columns="order_third", values="likelihood")
        wide = wide.dropna(subset=thirds)
        if len(wide) < 3:
            continue
        stat, p = friedmanchisquare(*(wide[third] for third in thirds))
        rows.append({class_col: cls, "statistic": stat, "p_value": p, "n_clutch_nights": len(wide)})
    return pd.DataFrame(rows)


# plot

def plot_likelihood_by_third(likelihood_df, colors, class_col, thirds, event_label, fname):
    classes = sorted(likelihood_df[class_col].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=likelihood_df, x="order_third", y="likelihood", hue=class_col,
                order=thirds, hue_order=classes, palette=colors, showfliers=False,
                width=0.5, linewidth=1, ax=ax)
    ax.axhline(1 / 3, color="red", linewidth=0.8, linestyle="--", label="chance (1/3)")
    ax.grid(False)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, classes + ["chance (1/3)"], fontsize=7, frameon=False, title=class_col)

    ax.set_xlabel(f"{event_label}-order third within clutch, per night")
    ax.set_ylabel("likelihood of falling in that third")
    ax.set_title(f"{event_label}")
    fig.tight_layout()

    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(os.path.join(FIGURES_DIR, fname + ext), dpi=150 if ext == ".png" else None,
                    bbox_inches="tight")
    print("saved", fname, "(.png/.svg/.pdf)")
    return fig, ax


def run(events, value_col, thirds, ascending, event_type, event_label):
    events = assign_order_third(events, value_col=value_col, thirds=thirds, ascending=ascending)

    for class_col in CLASS_COLS:
        # thirds are ranked over the whole clutch present that night, before dropping animals with unknown class for the class-level comparison 
        # - NEED TO FILL IN THE METADATA FOR THE MISSING ONES
        class_events = events.dropna(subset=[class_col])
        likelihood_df = class_likelihoods(class_events, class_col, thirds)
        likelihood_df.to_parquet(
            os.path.join(FIGURES_DIR, f"{event_type}_{class_col}_order_third_likelihood.parquet"))

        colors = class_colors(likelihood_df[class_col].unique())

        between = test_between_classes(likelihood_df, class_col, thirds).assign(test="between_classes")
        across = test_across_thirds(likelihood_df, class_col, thirds).assign(test="across_thirds")
        pd.concat([between, across], ignore_index=True).to_csv(
            os.path.join(FIGURES_DIR, f"{event_type}_{class_col}_order_third_stats.csv"), index=False)

        plot_likelihood_by_third(likelihood_df, colors, class_col, thirds, event_label,
                                  f"{event_type}_{class_col}_order_third_by_class")


def add_age_sex(df):
    df = df.copy()
    df["age_sex"] = np.where(df["age"].isna() | df["sex"].isna(), np.nan,
                              df["age"].astype(str) + "_" + df["sex"].astype(str))
    return df


if __name__ == "__main__":
    masterdf = preprocessing.load_regular_data()
    edge_events = analyses.build_edge_events_from_masterdf(masterdf)

    for event_type, event_label in (("sleep", "sleep-onset"), ("wake", "wake")):
        run(edge_events[edge_events["event_type"] == event_type].copy(),
            value_col="event_time", thirds=THIRDS, ascending=True,
            event_type=event_type, event_label=event_label)

    tst_events = add_age_sex(masterdf[["animal_id", "night_date", "clutch_id", "TST", "age", "sex"]])
    run(tst_events, value_col="TST", thirds=TST_THIRDS, ascending=True,
        event_type="tst", event_label="TST")
