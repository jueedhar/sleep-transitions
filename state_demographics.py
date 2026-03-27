# Juee Dhar

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
    plt.title("Proportion of group asleep across 24 hrs") 
    plt.legend() 
    plt.grid(True) 
    plt.show()

    return mean_asleep, ci95


def ind_transitions_pooled(dfs):
    """
    Compute sleep/wake transition probabilities for 30 min bins, pooled across all individuals and all days of the group.
    Plot starts at 9 PM UTC i.e. 00:00 Local time.
    """
    transitions = []

    for key, df in dfs.items():
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Previous state
        df['prev'] = df['pot_sleep'].shift(1)

        # sleep=1, wake=0
        df['sleep_to_wake'] = ((df['prev'] == 1) & (df['pot_sleep'] == 0)).astype(int)
        df['wake_to_sleep'] = ((df['prev'] == 0) & (df['pot_sleep'] == 1)).astype(int)

        # Shift day start to 9 PM UTC (so that it is 00:00 Local time)
        df['minute_of_day'] = ((df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute) - 21*60) % 1440

        # 30-min bin
        df['minute_bin'] = (df['minute_of_day'] // 30) * 30
        df['date'] = df['timestamp'].dt.date

        # normalized probabilities per bin
        for (date, minute_bin), g in df.groupby(['date','minute_bin']):
            time_in_sleep = (g['prev'] == 1).sum()
            time_in_wake  = (g['prev'] == 0).sum()

            p_sleep_to_wake = g['sleep_to_wake'].sum() / time_in_sleep if time_in_sleep > 0 else np.nan
            p_wake_to_sleep = g['wake_to_sleep'].sum() / time_in_wake if time_in_wake > 0 else np.nan

            stay_sleep = 1 - p_sleep_to_wake if not np.isnan(p_sleep_to_wake) else np.nan
            stay_wake  = 1 - p_wake_to_sleep if not np.isnan(p_wake_to_sleep) else np.nan

            transitions.append({
                "individual": key,
                "date": date,
                "minute_bin": minute_bin + 15,  # midpoint of 30-min bin
                "sleep_to_wake": p_sleep_to_wake,
                "wake_to_sleep": p_wake_to_sleep,
                "stay_sleep": stay_sleep,
                "stay_wake": stay_wake
            })

    df_bins = pd.DataFrame(transitions)

    # Pool across individuals AND days
    mean_sleep_to_wake = df_bins.groupby('minute_bin')['sleep_to_wake'].mean()
    se_sleep_to_wake   = df_bins.groupby('minute_bin')['sleep_to_wake'].sem()

    mean_wake_to_sleep = df_bins.groupby('minute_bin')['wake_to_sleep'].mean()
    se_wake_to_sleep   = df_bins.groupby('minute_bin')['wake_to_sleep'].sem()

    mean_stay_sleep = df_bins.groupby('minute_bin')['stay_sleep'].mean()
    se_stay_sleep   = df_bins.groupby('minute_bin')['stay_sleep'].sem()

    mean_stay_wake  = df_bins.groupby('minute_bin')['stay_wake'].mean()
    se_stay_wake    = df_bins.groupby('minute_bin')['stay_wake'].sem()

    ci_sleep_to_wake = 1.96 * se_sleep_to_wake
    ci_wake_to_sleep = 1.96 * se_wake_to_sleep
    ci_stay_sleep    = 1.96 * se_stay_sleep
    ci_stay_wake     = 1.96 * se_stay_wake

    minute_bins = sorted(mean_sleep_to_wake.index)

    # Convert minute_bin to HH:MM for x-axis
    labels = [f"{int((m+1260)%1440//60):02d}:{int((m+1260)%60):02d}" for m in minute_bins]

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].errorbar(minute_bins, mean_sleep_to_wake.loc[minute_bins], yerr=ci_sleep_to_wake.loc[minute_bins],
                        marker='o', color='blue', capsize=4)
    axes[0].set_ylabel("Probability per minute")
    axes[0].set_title("Sleep → Wake (1 → 0)")
    axes[0].set_xticks(minute_bins[::2])  # show every hour or every 2 bins
    axes[0].set_xticklabels(labels[::2], rotation=45)


    axes[1].errorbar(minute_bins, mean_wake_to_sleep.loc[minute_bins], yerr=ci_wake_to_sleep.loc[minute_bins],
                        marker='o', color='orange', capsize=4)
    axes[1].set_ylabel("Probability per minute")
    axes[1].set_title("Wake → Sleep (0 → 1)")
    axes[1].set_xticks(minute_bins[::2])  
    axes[1].set_xticklabels(labels[::2], rotation=45)


    axes[2].errorbar(minute_bins, mean_stay_sleep.loc[minute_bins], yerr=ci_stay_sleep.loc[minute_bins],
                        marker='o', color='blue', capsize=4, label="Sleep → Sleep")
    axes[2].errorbar(minute_bins, mean_stay_wake.loc[minute_bins], yerr=ci_stay_wake.loc[minute_bins],
                        marker='s', color='orange', capsize=4, label="Wake → Wake")
    axes[2].set_ylabel("Probability per minute")
    axes[2].set_xlabel("Time of Day (9 PM → 9 PM)")
    axes[2].set_title("Staying in same state")
    axes[2].legend()

    axes[2].set_xticks(minute_bins[::2])  
    axes[2].set_xticklabels(labels[::2], rotation=45)

    plt.tight_layout()
    plt.show()
    return df_bins


if __name__ == "__main__":
    group_id_input = "your_group_id_here"
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

    # Run the pooled sleep function and individual sleep transit function
    if n_dfs > 0:
        mean_prop, ci95 = group_sleep_prob_pooled(dfs)
        df_bins = ind_transitions_pooled(dfs)
    else:
        print("No valid dataframes to process.")

