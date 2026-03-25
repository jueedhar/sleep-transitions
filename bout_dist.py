import pandas as pd
import numpy as np
import powerlaw
import inactivity_parquet_load


def bout_data(series, state=1):
    """
    Compute the number of opposite values between consecutive states.
    - state=1 -> counts 0s between 1s (activity bouts)
    - state=0 -> counts 1s between 0s (inactivity bouts)
    Returns a list of counts.
    """
    counts = []
    current_count = 0
    started = False

    for val in series:
        if val == state:
            if started:
                counts.append(current_count)
            current_count = 0
            started = True
        else:
            if started:
                current_count += 1
    return counts

def bout_df(counts):
    """
    Converts a list of bout counts into a frequency DataFrame.
    """
    df = pd.DataFrame(counts, columns=['bout_length'])
    freq_df = df['bout_length'].value_counts().reset_index()
    freq_df.columns = ['bout_length', 'frequency']
    freq_df = freq_df.sort_values('bout_length').reset_index(drop=True)
    return freq_df

