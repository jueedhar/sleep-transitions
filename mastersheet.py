import os
import pandas as pd


parquet_folder = "D:/2025_data_baboons/inactivity"
metadata_file = "D:/2025_data_baboons/Baboons MBRP Mpala Kenya-reference-data.csv"
output_csv = "MASTER-DATA-SHEET.csv"


metadata = pd.read_csv(metadata_file)

meta_row = {
    "date": , 
    "ind": ,
    "t_wake": ,
    "t_sleep": ,
    "group_id": ,
    "age": ,
    "sex": 
}

# Read all parquet files
dfs = []
parquet_files = [f for f in os.listdir(parquet_folder) if f.endswith(".parquet")]

for file in parquet_files:
    animal_id = os.path.splitext(file)[0]
    parquet_path = os.path.join(parquet_folder, file)
    
    try:
        df = pd.read_parquet(parquet_path)
        
        # Merge metadata if animal-id exists
        meta_row_animal = metadata[metadata['animal-id'] == animal_id]
        if not meta_row_animal.empty:
            df['group_id'] = meta_row_animal['animal-group-id'].values[0]
            df['age'] = meta_row_animal['age'].values[0]
            df['sex'] = meta_row_animal['sex'].values[0]
        else:
            df['group_id'] = None
            df['age'] = None
            df['sex'] = None
        
        dfs.append(df)
        print(f"Loaded {animal_id}")
    except Exception as e:
        print(f"Failed to read {file}: {e}")

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
else:
    combined_df = pd.DataFrame(columns=["date", "ind", "t_wake", "t_sleep", "group_id", "age", "sex"])

# Add meta row on top
combined_df = pd.concat([pd.DataFrame([meta_row]), combined_df], ignore_index=True)

# Reorder columns
#combined_df = combined_df[["date", "ind", "t_wake", "t_sleep", "group_id", "age", "sex"]]

# Save CSV
combined_df.to_csv(output_csv, index=False)
print(f"Combined CSV saved to: {output_csv}")
