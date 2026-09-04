# Juee Dhar
# 02 Sept 2026

""" Diagnostic only - does not change control_sims.py. """

import numpy as np
import pandas as pd

import analyses
import control_sims
import preprocessing

N_SHUFFLES = 10
SEEDS = range(N_SHUFFLES) 


def count_collisions(date_map, clutch_id):
    """
    date_map: output of control_sims.make_date_map (animal_id, night_date,new_night_date)

    Collision = a (clutch_id, night_date) group where 2+ animals share the same new_night_date. Returns the count of such groups, and the
    count of individual animal-nights involved (a group of 3 colliding counts as 1 group but 3 animal-nights)
    """
    df = date_map.copy()
    df["clutch_id"] = df["animal_id"].map(clutch_id)

    n_groups = 0
    n_animal_nights = 0
    for _, sub in df.groupby(["clutch_id", "night_date"]):
        if len(sub) < 2:
            continue
        dup_counts = sub["new_night_date"].value_counts()
        dup_counts = dup_counts[dup_counts > 1]
        if len(dup_counts):
            n_groups += len(dup_counts)
            n_animal_nights += int(dup_counts.sum())

    return n_groups, n_animal_nights


if __name__ == "__main__":
    masterdf = preprocessing.load_regular_data()
    edge_events = analyses.build_edge_events_from_masterdf(masterdf)

    clutch_id = (edge_events[["animal_id", "clutch_id"]].drop_duplicates("animal_id").set_index("animal_id")["clutch_id"])
    total_animal_nights = len(edge_events[["animal_id", "night_date"]].drop_duplicates())

    print(f"{edge_events['animal_id'].nunique()} animals, "
          f"{edge_events['night_date'].nunique()} distinct nights, "
          f"{clutch_id.nunique()} clutches, "
          f"{total_animal_nights} total animal-nights\n")

    #for the reshuffles
    results = []
    for seed in SEEDS: 
        date_map = control_sims.make_date_map(edge_events, seed=seed)
        n_groups, n_animal_nights = count_collisions(date_map, clutch_id)
        pct = 100 * n_animal_nights / total_animal_nights
        results.append((seed, n_groups, n_animal_nights, pct))
        print(f"seed {seed}: {n_groups} colliding (clutch, night) group(s), "
              f"{n_animal_nights}/{total_animal_nights} animal-nights affected ({pct:.2f}%)")

    n_groups_all = np.array([r[1] for r in results])
    n_an_all = np.array([r[2] for r in results])
    pct_all = np.array([r[3] for r in results])
    print(f"\nacross {N_SHUFFLES} shuffles: "
          f"groups mean={n_groups_all.mean():.1f} (min={n_groups_all.min()}, max={n_groups_all.max()}), "
          f"animal-nights mean={n_an_all.mean():.1f}/{total_animal_nights} "
          f"({pct_all.mean():.2f}%, min={pct_all.min():.2f}%, max={pct_all.max():.2f}%)")

    