#!pip install powerlaw

import os.path

import pandas as pd
import powerlaw
import matplotlib.pyplot as plt

import config
from inactivity_parquet_load import get_parquet_files_for_group, read_parquets_to_dfs  

BASE_DIR = config.PROJECTROOT
metadata_path = os.path.join(config.DATA, "Baboons-MBRP-Mpala-Kenya-reference-data.csv")
base_inactivity_path =\
"/media/pranav/MPI_Dirs/EAS_shared/baboon/working/data/processed/2025/acc/inactivity"

def inter_bout_intervals(series, target=0):
    """
    Compute inter-bout intervals (counts between target events,so a target = 0 computes
    the inactivity bout by counting no. of 1s between the 0s).
      """
    intervals = []
    count = None
    for val in series:
        if val == target:
            if count is not None:
                intervals.append(count)
            count = 0
        else:
            if count is not None:
                count += 1
    return intervals

def bout_df(counts):
    """
    Convert list of counts to frequency dataframe.
    """
    df = pd.DataFrame(counts, columns=['bout_length'])
    freq_df = df['bout_length'].value_counts().reset_index()
    freq_df.columns = ['bout_length', 'frequency']
    return freq_df.sort_values('bout_length').reset_index(drop=True)


# Powerlaw fitting & plotting

def fit_and_plot_IBI(data, series_name="IBI Series"):
    """
    This function fits standard distributions to IBI data. The truncated powerlaw outperformed the power law,
    lognormal, stretched exponential and of course exponential, hence sthe comparisions are done wrt it.
    """
    
    data = [x for x in data if x > 0]


    fit = powerlaw.Fit(data, discrete=True)

    # Truncated power law
    tpl = fit.truncated_power_law
    alpha = tpl.alpha
    lambda_ = tpl.parameter1  # exponential cutoff
    xmin = tpl.xmin

    print(f"\nTruncated Power Law fit for {series_name}:")
    print(f"alpha: {alpha:.5f}")
    print(f"lambda (cutoff): {lambda_:.5f}")
    print(f"xmin: {xmin}\n")

    # distribution comparisons
    comparisons1 = [
        ('power_law', 'exponential'),
        ('power_law', 'lognormal'),
        ('stretched_exponential', 'exponential'),
        ('lognormal', 'exponential')
    ]
    for d1, d2 in comparisons1:
        R, p = fit.distribution_compare(d1, d2)
        print(f"{d1} vs {d2} → R={R:.3f}, p={p:.5f}")

    comparisons2 = [
        ('truncated_power_law', 'power_law'),
        ('truncated_power_law', 'lognormal'),
        ('truncated_power_law', 'exponential'),
        ('truncated_power_law', 'stretched_exponential')
    ]
    for d1, d2 in comparisons2:
        R, p = fit.distribution_compare(d1, d2)
        print(f"{d1} vs {d2} → R={R:.3f}, p={p:.5f}")



    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    fit.plot_ccdf(label='Data', linewidth=2, color='black', ax=ax)
    fit.power_law.plot_ccdf(label='Power law', ax=ax)
    fit.truncated_power_law.plot_ccdf(label='Truncated Power Law', ax=ax)
    fit.lognormal.plot_ccdf(label='Lognormal', ax=ax)
    fit.stretched_exponential.plot_ccdf(label='Weibull', ax=ax)
    fit.exponential.plot_ccdf(label='Exponential', ax=ax)
    ax.legend()
    ax.set_xlabel('Inter-bout interval')
    ax.set_ylabel('CCDF')
    ax.set_ylim(1e-5, 1)
    ax.set_title(f'Distribution comparison for {series_name}')
    plt.show()

    return fig, ax


if __name__ == "__main__":
    metadata = pd.read_csv(metadata_path)
    group_id_input = "Lapis"
    parquet_files = get_parquet_files_for_group(metadata, group_id_input, base_path=base_inactivity_path)
    dfs = read_parquets_to_dfs(parquet_files)

    assigned_dfs = []
    for df_name, df_obj in dfs.items():
        if df_obj is not None:
            assigned_dfs.append(df_name)

    # Compute IBIs
    for df_name in assigned_dfs:
        df = dfs[df_name]['df']
        if 'pot_sleep' not in df.columns:
            print(f"{df_name} has no 'pot_sleep' column, skipping.")
            continue

        print(f"\nProcessing {df_name}...")
        counts = inter_bout_intervals(df['pot_sleep'])
        freq = bout_df(counts)

        # Fit distributions and plot
        fig, ax = fit_and_plot_IBI(counts, series_name=df_name)
        plt.close(fig)
        plt.cla()
