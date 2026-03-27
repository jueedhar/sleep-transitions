# Juee Dhar

import os
import pandas as pd
import config

BASE_DIR = config.PROJECTROOT
metadata_path = os.path.join(BASE_DIR, "Baboons MBRP Mpala Kenya-reference-data.csv")
base_inactivity_path = os.path.join(BASE_DIR, "inactivity")

def get_parquet_files_for_group(metadata, group_id, base_path=base_inactivity_path):
    
    """
    This function extracts all parquet files corresponding to animal-id pulled from the Baboons MBRP Mpala Kenya-reference-data.csv for a single group_id 
    """
    
    animal_ids = metadata.loc[metadata['animal-group-id'] == group_id,'animal-id'].tolist()
    
    # check for missing files
    available_files = {}
    missing_ids = []
    
    for animal_id in animal_ids:
        parquet_path = os.path.join(base_path, f"{animal_id}.parquet")
        
        if os.path.exists(parquet_path):
            available_files[animal_id] = parquet_path
        else:
            missing_ids.append(animal_id)
    
    # print status
    if missing_ids:
        print(f"Missing parquet files for animal IDs: {missing_ids}")
    else:
        print("All parquet files are available for this group.")
    
    return available_files


def read_parquets_to_dfs(parquet_dict):
     
    """
    This function saves all parquet files corresponding to a single group_id to a dictionary, where different individuals are loaded as df1, df2 and onwards.
    """
     
    dfs = {}
    
    for idx, (animal_id, file) in enumerate(parquet_dict.items(), start=1):
        try:
            df = pd.read_parquet(file)
            dfs[f"df{idx}"] = {"animal_id": animal_id,"df": df}
            print(f"Loaded {animal_id} into df{idx} ({file})")
        except Exception as e:
            print(f"Failed to read {file}: {e}")
    
    return dfs


if __name__ == "__main__":
    metadata = pd.read_csv(metadata_path)
    group_id_input = "Chartreuse" #change only this to load data for a particular group

    parquet_files = get_parquet_files_for_group(metadata, group_id_input,base_path=base_inactivity_path) # FIXED: pass metadata
    dfs = read_parquets_to_dfs(parquet_files)

    assigned_dfs = []

    for df_name, df_obj in dfs.items():
        if df_obj is not None:
            assigned_dfs.append(df_name)
            print(f"{df_name} assigned, shape: {df_obj.shape}")
        else:
            print(f"{df_name} is None")

    n_dfs = len(assigned_dfs)
    print(f"\nTotal DataFrames assigned: {n_dfs}")

    metadata = pd.read_csv(metadata_path)
    group_id_input = "Chartreuse"

    
