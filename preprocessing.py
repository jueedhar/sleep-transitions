#Juee Dhar
#17.08.2026

import os

import pandas as pd

import config
import populate_mastersheet


TST_THRESHOLD = 300          # TST < this => "disturbed", else "regular"
MIN_INDIVIDUALS_PER_CLUTCH = 5   # per clutch_id PER NIGHT
LOAD_STATUS = "regular"      # as opposed to disturbed nights
TST_CSV_PATH = os.path.join(config.DATA, "combined_sleep_analysis.csv")



def mask_and_filter(masterdf,
                                     tst_csv_path=TST_CSV_PATH,
                                     tst_threshold=TST_THRESHOLD,
                                     min_individuals_per_clutch=MIN_INDIVIDUALS_PER_CLUTCH):
    """
    Takes masterdf from populate_mastersheet.generate_master_sheet()
    and does in order:

    1. Merges in TST from combined_sleep_analysis.csv (columns: tag,
       night_date, TST) and adds a 'disturbance_status' column: 'disturbed'
       if TST < tst_threshold, else 'regular'. Rows with no TST match get
       NaN status (reported, not silently dropped here).
    2. Drops clutch-nights (a specific clutch_id on a specific night_date)
       that have fewer than min_individuals_per_clutch unique animal_id
       that night. This is evaluated per clutch PER NIGHT, not on the
       clutch's total roster -- a clutch can pass on some nights and fail
       on others. No minimum-number-of-nights requirement anywhere.

    Returns masterdf with the new 'disturbance_status' column, filtered to
    only the clutch-nights that pass the individual-count minimum.
    """
    masterdf = masterdf.rename(columns={"ind": "animal_id", "date": "night_date"})

    tst_df = pd.read_csv(tst_csv_path)[["tag", "night_date", "TST"]]
    tst_df = tst_df.rename(columns={"tag": "animal_id"})
    tst_df["night_date"] = pd.to_datetime(tst_df["night_date"])
    masterdf = masterdf.merge(tst_df, on=["animal_id", "night_date"], how="left")

    n_unmatched = masterdf["TST"].isna().sum()
    if n_unmatched:
        print(f"mask_and_filter: {n_unmatched} row(s) had no TST match "
              f"in {tst_csv_path} -- their disturbance_status is NaN")

    masterdf["disturbance_status"] = "regular"
    masterdf.loc[masterdf["TST"] < tst_threshold, "disturbance_status"] = "disturbed"
    masterdf.loc[masterdf["TST"].isna(), "disturbance_status"] = pd.NA
    
    #n_individuals_that_night = masterdf.groupby(
    #    ["clutch_id", "night_date"])["animal_id"].transform("nunique")
    #passes_clutch_minimum = n_individuals_that_night >= min_individuals_per_clutch
 
    passes_clutch_minimum = masterdf["clutch_size"] >= min_individuals_per_clutch

    n_dropped = (~passes_clutch_minimum).sum()
    if n_dropped:
        print(f"mask_and_filter: dropping {n_dropped} row(s) from "
              f"clutch-nights with fewer than {min_individuals_per_clutch} individuals")

    return masterdf[passes_clutch_minimum].reset_index(drop=True)


def load_regular_data(status=LOAD_STATUS,
                       tst_csv_path=TST_CSV_PATH,
                       tst_threshold=TST_THRESHOLD,
                       min_individuals_per_clutch=MIN_INDIVIDUALS_PER_CLUTCH):
    """
    populate_mastersheet.generate_master_sheet() ->
    add_disturbance_mask_and_filter, and returns only rows whose
    disturbance_status == status 
    Call this instead of populate_mastersheet.generate_master_sheet() directly.
    """
    masterdf = populate_mastersheet.generate_master_sheet()
    masterdf = mask_and_filter(
        masterdf,
        tst_csv_path=tst_csv_path,
        tst_threshold=tst_threshold,
        min_individuals_per_clutch=min_individuals_per_clutch,
    )
    masterdf = masterdf[masterdf["disturbance_status"] == status].reset_index(drop=True)
    print(f"load_regular_data: {masterdf['animal_id'].nunique()} animals, "
          f"{masterdf['night_date'].nunique()} nights remaining with status == '{status}'")
    return masterdf


if __name__ == "__main__":
    df = load_regular_data()
    print(df.shape)


