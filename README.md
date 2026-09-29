Sleep/wake transition-rate analysis for baboon clutches: real data vs.
control_sims-shuffled null, for both edge (single onset/wake per animal per
night) and bulk (repeated in-night transitions) events.

## Core pipeline (data-flow order)

- **config.py** -- paths (`DATA`, `FIGURES`), percentile-bin thresholds, output formats.
- **utilities.py** -- `saveimg` (writes a figure to every format in `config.formats`), `sprint`.
- **populate_mastersheet.py** -- builds the master data sheet: animal-night sleep/wake
  edge times merged with reference metadata, cluster-based sleep-site labels, and
  group demographics (size/coverage class). Adds `clutch_size` and `wake_site_type`
  (previous real night's `sleep_site_type`).
- **preprocessing.py** -- `load_regular_data()`, the standard entry point for masterdf.
  Filters to nights with both sleep and wake edge times, `disturbance_status == "regular"`
  (TST above threshold), and `clutch_size >= MIN_INDIVIDUALS_PER_CLUTCH` counted *within*
  that status.
- **durations.py** -- the two at-risk models: `get_transition_duration_table_edge`
  (one onset/wake per animal per night, monotonically shrinking pool) and
  `get_transition_duration_tables_bulk` (repeated in-night transitions, live
  non-monotonic occupancy pool; caps `interval_dur` at `BULK_MAX_INTERVAL_DUR_SEC`).
- **estimation.py** -- fits the per-cell exponential rate, bootstraps its SE, corrects
  for at-risk pool size, and aggregates across pool sizes into one `p_estimate` per
  percentile bin.
- **analyses.py** -- builds edge/bulk event streams from masterdf and the raw
  per-animal inactivity parquets (`build_edge_events_from_masterdf`, `build_bulk_events`,
  `split_edge_bulk_events` -- enforces the core-sleep window and per-animal edge
  exclusion); wraps durations.py + estimation.py (`build_duration_tables`,
  `compute_estimates`); all plotting (`plot_eventtype_panels`, `plot_category_panels`,
  `plot_bulk_interval_duration`).
- **control_sims.py** -- the real null: `make_date_map`/`apply_date_map` independently
  permute each animal's own real `night_date`s (event_time shifted to match, so
  time-of-night is preserved exactly); `filter_min_clutch_size` re-checks the size
  floor after shuffling.
- **main.py** -- the entry point. Loads data, builds edge/bulk events, runs both real
  and control_sims-shuffled estimates across every `BY_DIMENSIONS` split (age, sex,
  size_class, coverage_class, sleep/wake site type, night_third), saves parquets to
  `Data/prop_outputs` and plots to `Figures`.


## Testing utilities (synthetic data)

- **simulations.py** / **runsims.py** -- generate synthetic wake/sleep tables and run
  them through the estimator, to test the estimation machinery independent of real data.


