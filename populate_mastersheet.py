# Juee Dhar

import os

import pandas as pd 
import pyarrow as pa

import config

'''
This creates a master data sheet from 4 different data sheets, the MRBP Mpala Kenya reference data.csv, cluster_labels.csv and 
individual_night_locations.csv, combine_sleep_analysis.csv, later two of which are updated daily.
'''

def generate_master_sheet():
# where the script is located
    BASE_DIR = config.PROJECTROOT

# Data directory (must be a folder named "Data" next to the script)
    DATA_DIR = config.DATA
# Input files
    sleep_df = pd.read_csv(
        os.path.join(DATA_DIR, 'combined_sleep_analysis.csv'),
        usecols=['tag', 'night_date', 'onset', 'waking']
    )

# Rename 'tag' to 'animal_id' for consistent merging
    sleep_df.rename(columns={'tag': 'animal_id'}, inplace=True)

    locations_df = pd.read_csv(
        os.path.join(DATA_DIR, 'individual_night_locations.csv'),
        usecols=['animal_id', 'cluster_united', 'group_id', 'date']
    )
    # Rename 'night_date' to 'date' to match sleep_df
    locations_df.rename(columns={'date': 'night_date'}, inplace=True)

    # Merge both on 'animal_id' and 'date' (this caused the problem with the repeats!)
    merged_df = sleep_df.merge(
        locations_df,
        on=['animal_id', 'night_date'],
        how='left'
    )

# Load cluster labels  to add sleep_site_type
    cluster_labels_df = pd.read_csv(
        os.path.join(DATA_DIR, 'cluster_labels.csv'),
        usecols=['cluster_united', 'sleep_site_type']
    )

    merged_df = merged_df.merge(
        cluster_labels_df,
        on='cluster_united',
        how='left'
    )

    reference_df = pd.read_csv(
        os.path.join(DATA_DIR, 'Baboons-MBRP-Mpala-Kenya-reference-data.csv'),
        usecols=['animal-id', 'animal-comments', 'animal-sex']
    )
    reference_df.rename(columns={'animal-id': 'animal_id'}, inplace=True)


    final_df = merged_df.merge(
        reference_df,
        on= 'animal_id',  # now all merges use the same column
        how='left'
    )
# Rename columns 
    final_df.rename(columns={
        'animal_id': 'ind',
        'night_date' : 'date',
        'onset': 't_sleep',
        'waking': 't_wake',
        'animal-comments':'age',
        'animal-sex': 'sex',
        'cluster_united': 'clutch_id',
    }, inplace=True)

    # Ensure proper sorting for lag operation
    final_df = final_df.sort_values(['ind', 'date'])

    # wake_site_type = previous day's sleep_site_type
    final_df['wake_site_type'] = final_df.groupby('ind')['sleep_site_type'].shift(1)

    # clutch_size = number of individuals in same clutch (per date)
    final_df['clutch_size'] = final_df.groupby(['date', 'clutch_id'])['ind'].transform('nunique')

    final_df.date = pd.to_datetime(final_df.date)
    final_df.t_sleep = pd.to_datetime(final_df.t_sleep)
    final_df.t_wake = pd.to_datetime(final_df.t_wake)
    return final_df

# Save
def save_master_df(df):
    df.to_parquet(
        config.MASTER_DATA_SHEET,
        index=False
    )

if __name__ == "__main__":
    master_df = generate_master_sheet()
    save_master_df(master_df)
