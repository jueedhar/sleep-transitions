# Juee Dhar, 28 August 2026
# Pranav Minasandra, 27 Mar 2026

import os

import matplotlib.pyplot as plt
import seaborn as sns

import analyses
import config
import control_sims
import preprocessing
import utilities

sns.set_theme(style="white", palette="husl")

GROUP_COL = "group_id"

# One real group_id (e.g. "Emerald") for a quick test run, or None for everything.
TEST_GROUP_ID = None

CONTROL = True
CONTROL_SEED = 42

N_BOOT = 20

OVERALL = True
BY_DIMENSIONS = ["age", "sex", "age_sex", "size_class", "sleep_site_type", "wake_site_type", "night_third"]

RUN_EDGE = True
RUN_BULK = True

# (tag, dimension) -> y-axis limits, for the weird plots
YLIMS = {("edge", "age"): (0, 0.005)}


def _draw(est, ctrl_est, plot_fn, name, ylim):
    fig, axes = plot_fn(est)
    if ctrl_est is not None and not ctrl_est.empty:
        plot_fn(ctrl_est, axes=axes, linestyle="--", alpha=0.45,
                suffix=" (control)", set_titles=False)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(*ylim)
    utilities.saveimg(fig, name)
    plt.close(fig)


def run(events_df, control_events, tag, output_dir):
    # Built once per tag, then reused by every dimension below.
    tables = analyses.build_duration_tables(events_df, group_col=GROUP_COL)
    ctrl_tables = (analyses.build_duration_tables(control_events, group_col=GROUP_COL)
                   if control_events is not None else None)

    jobs = [("overall", "none")] if OVERALL else []
    jobs += [(dim, dim) for dim in BY_DIMENSIONS
             if dim != "night_third" or tag == "bulk"]

    for name, by in jobs:
        est = analyses.compute_estimates(tables, by=by, n_boot=N_BOOT)
        if est.empty:
            print(f"{tag}/{name}: no estimates, skipped")
            continue
        est.to_parquet(os.path.join(output_dir, f"{tag}_{name}.parquet"))

        ctrl_est = None
        if ctrl_tables is not None:
            ctrl_est = analyses.compute_estimates(ctrl_tables, by=by, n_boot=N_BOOT)
            if not ctrl_est.empty:
                ctrl_est.to_parquet(os.path.join(output_dir, f"{tag}_{name}_control.parquet"))

        ylim = YLIMS.get((tag, name))
        _draw(est, ctrl_est, analyses.plot_category_panels,
              f"{tag}_{name}_category_panels", ylim)
        if by != "none":
            _draw(est, ctrl_est, analyses.plot_eventtype_panels,
                  f"{tag}_{name}_eventtype_panels", ylim)

        print(f"{tag}/{name}: done")


if __name__ == "__main__":
    output_dir = os.path.join(config.DATA, "prop_outputs")
    os.makedirs(output_dir, exist_ok=True)

    masterdf = preprocessing.load_regular_data()

    if TEST_GROUP_ID is not None:
        masterdf = masterdf[masterdf[GROUP_COL] == TEST_GROUP_ID].reset_index(drop=True)
        print(f"TEST RUN: {GROUP_COL} == {TEST_GROUP_ID} only -- "
              f"{masterdf['animal_id'].nunique()} animals, {masterdf['night_date'].nunique()} nights")

    edge_events = analyses.build_edge_events_from_masterdf(masterdf)

    # One shuffle for the whole session, so edge and bulk get the same relabelling.
    date_map = control_sims.make_date_map(edge_events, seed=CONTROL_SEED) if CONTROL else None

    if RUN_EDGE:
        control_events = control_sims.apply_date_map(edge_events, date_map) if CONTROL else None
        run(edge_events, control_events, "edge", output_dir)
        del control_events

    if RUN_BULK:
        bulk_events = analyses.assign_night_third(
            analyses.build_bulk_events(masterdf, edge_events))
        control_events = control_sims.apply_date_map(bulk_events, date_map) if CONTROL else None
        run(bulk_events, control_events, "bulk", output_dir)