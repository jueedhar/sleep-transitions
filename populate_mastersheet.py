import os
import pandas as pd 
import pyarrow as pa

'''
This creates a master data sheet from 4 different data sheets, the MRBP Mpala Kenya reference data.csv, cluster_labels.csv and 
individual_night_locations.csv, combine_sleep_analysis.csv, later two of which are updated daily.
'''
# where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory (must be a folder named "Data" next to the script)
DATA_DIR = os.path.join(BASE_DIR, "Data")
# Input files
sleep_df = pd.read_csv(
    os.path.join(DATA_DIR, 'combined_sleep_analysis.csv'),
    usecols=['tag', 'night_date', 'onset', 'waking']
)

# Rename 'tag' to 'animal_id' for consistent merging
sleep_df.rename(columns={'tag': 'animal_id'}, inplace=True)

locations_df = pd.read_csv(
    DATA_DIR, 'individual_night_locations.csv',
    usecols=['animal_id', 'cluster_united', 'group_id']
)

# Merge sleep with locations on 'animal_id'
merged_df = sleep_df.merge(
    locations_df,
    on='animal_id',
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
    on='animal_id',  # now all merges use the same column
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

# Save
final_df.to_parquet(
    os.path.join(DATA_DIR, 'masterdata.parquet'),
    index=False
)

final_df.to_csv(
    os.path.join(DATA_DIR, 'masterdata.csv'),
    index=False
)

print("Merged metadata saved as masterdata.parquet and masterdata.csv")