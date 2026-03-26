import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inactivity_parquet_load

def group_sleep_prob_pooled(dfs):
    """
    This function calculates and plots the proportion of collars awake at every point of the the day.
    Potential issue - If no overlapping window for ALL read collars, it doesn't proceed. 
    """

    # find overlapping timestamp window
    start_times = [df['timestamp'].min() for df in dfs.values()]
    end_times = [df['timestamp'].max() for df in dfs.values()]
    overlap_start = max(start_times)
    overlap_end = min(end_times)
    if overlap_start >= overlap_end:
        raise ValueError("No overlapping timestamp range")
    print(f"Overlapping period: {overlap_start} → {overlap_end}")

    # create per-df table with minute-of-day and pot_sleep
    df_list = []

    for key, df in dfs.items():
        df = df.copy()
        df = df[(df['timestamp'] >= overlap_start) & (df['timestamp'] <= overlap_end)]
        if df.empty:
            continue

        # Shift day start to 9 PM UTC (so that it is 00:00 Local time)
        df['timestamp_local'] = df['timestamp'] + pd.Timedelta(hours=3)
        df['date'] = df['timestamp_local'].dt.floor('D')
        #df['minute_of_day'] = df['timestamp_local'].dt.hour * 60 + df['timestamp_local'].dt.minute
        df['minute_of_day'] = ((df['timestamp_local'].dt.hour * 60 + df['timestamp_local'].dt.minute) - 1260) % 1440

        # Error fix -  dtypes are consistent
        df['minute_of_day'] = df['minute_of_day'].astype('int64')
        df['date'] = pd.to_datetime(df['date'])

        df = df[['date','minute_of_day','pot_sleep']].copy()
        df = df.rename(columns={'pot_sleep': key})  # column named by df key
        df_list.append(df)

    # merge all dfs on date + minute_of_day
    from functools import reduce
    merged = reduce(lambda left,right: pd.merge(left, right, on=['date','minute_of_day'], how='outer'), df_list)

    merged = merged.fillna(np.nan) #unneccesary, just making sure it is not read as 0

    # sum across dfs to get number asleep at that minute
    df_keys = list(dfs.keys())
    num_dfs = len(dfs)  # total number of dataframes
    merged['num_asleep'] = merged[df_keys].sum(axis=1)/ num_dfs
    #time_of_day = (pd.to_timedelta((merged['minute_of_day'] + 180) % 1440, unit='m')).astype(str)

    # compute mean and 95% CI per min of day across days
    grouped = merged.groupby('minute_of_day')['num_asleep']
    mean_asleep = grouped.mean()
    se_asleep = grouped.sem()
    ci95 = 1.96 * se_asleep

    plt.figure(figsize=(12,4))
    plt.plot(mean_asleep.index, mean_asleep.values, color='blue', label='Mean # asleep') 
    plt.fill_between(mean_asleep.index, mean_asleep - ci95, mean_asleep + ci95, color='blue', alpha=0.2, label='95% CI') 
    plt.xlabel("Minute of Day (starts at 00:00 Local Time (i.e. UTC +3))") 
    plt.ylabel("Proportion of collars asleep") 
    plt.ylim(0,1) 
    plt.title("Propotion of group asleep across 24 hrs") 
    plt.legend() 
    plt.grid(True) 
    plt.show()

    return mean_asleep, ci95

if __name__ == "__main__":
    parquet_files = inactivity_parquet_load.get_parquet_files_for_group(group_id_input)
    dfs = inactivity_parquet_load.read_parquets_to_dfs(parquet_files)

    assigned_dfs = []

    for df_name, df_obj in dfs.items():
        if df_obj is not None:
            assigned_dfs.append(df_name)
            print(f"{df_name} assigned, shape: {df_obj.shape}")
        else:
            print(f"{df_name} is None")

    n_dfs = len(assigned_dfs)
    print(f"\nTotal DataFrames assigned: {n_dfs}")

    # Run the pooled sleep function
    if n_dfs > 0:
        mean_prop, ci95 = group_sleep_prob_pooled(dfs)
    else:
        print("No valid dataframes to process.")

