# Pranav Minasandra
# 27 Mar 2026
# pminasandra.github.io

import os
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import analyses
import config
import populate_mastersheet
import utilities

sns.set_theme(style="white", palette="husl")

# FLOW CONTROL
OVERALL_ANALYSIS    = True
BY_SITE_TYPE        = True
BY_AGE_SEX          = True

if __name__ == "__main__":
# First, make the necessary folders
    output_dir = os.path.join(config.DATA, "prop_outputs")
    os.makedirs(output_dir, exist_ok=True)

# Create the master dataframe
    masterdf = populate_mastersheet.generate_master_sheet()


# Analysis 1: Are sleep and wake social dynamics symmetric?
    if OVERALL_ANALYSIS:
        results, fig, ax = analyses.analyse_sleep_wake_asymmetry_by(masterdf,
                                by="none",
                                foreach="none")

        outname = os.path.join(output_dir, "overall_percentiles.parquet")
        results.to_parquet(outname)
        utilities.saveimg(fig, "real_transition_probabilities")
        plt.close(fig)
        plt.clf()
        plt.cla()


# Analysis 2: How do sleep/wake social dynamics differ across different sleep-sites?
    if BY_SITE_TYPE:
        results, fig, ax = analyses.analyse_sleep_wake_asymmetry_by(masterdf,
                        by="sleep_site_type",
                        foreach="none")
        results = results[results.eventtype == 'sleep']
        outname = os.path.join(output_dir, "sleep_probs_by_sleep_site_type.parquet")
        results.to_parquet(outname)
        utilities.saveimg(fig, "sleep_dynamics_by_site_type")
        plt.close(fig)
        plt.clf()
        plt.cla()


        results, fig, ax = analyses.analyse_sleep_wake_asymmetry_by(masterdf,
                        by="wake_site_type",
                        foreach="none")
        results = results[results.eventtype == 'wake']
        outname = os.path.join(output_dir, "wake_probs_by_wake_site_type.parquet")
        results.to_parquet(outname)
        utilities.saveimg(fig, "wake_dynamics_by_wake_type")
        plt.close(fig)
        plt.clf()
        plt.cla()

# Analysis 4: How are sleep/wake dynamics affected by age-sex class
    if BY_AGE_SEX:
        mdf = masterdf.copy()
        mdf['age-sex'] = mdf['age'] + " " + mdf['sex']
        mdf.dropna(inplace = True)

        results, fig, ax = analyses.analyse_sleep_wake_asymmetry_by(mdf,
                                by="none",
                                foreach="age-sex")
        outname = os.path.join(output_dir, "probs_by_age_sex_class.parquet")
        results.to_parquet(outname)
        utilities.saveimg(fig, "transition_dynamics_by_age_sex_class")
        plt.close(fig)
        plt.clf()
        plt.cla()

