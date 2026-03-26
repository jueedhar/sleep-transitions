
# Pranav Minasandra
# pminasandra.github.io
# March 23, 2026

import os
import os.path

import numpy as np

#Directories
PROJECTROOT = open(".cw", "r").read().rstrip()
DATA = os.path.join(PROJECTROOT, "Data")
FIGURES = os.path.join(PROJECTROOT, "Figures")

MASTER_DATA_SHEET = os.path.join(DATA, "baboon_sleep_wake_transitions.parquet")

formats=['png', 'pdf', 'svg']

# Simulation details

PERCENTILE_THRESHOLDS = np.linspace(0.1, 1, 9).tolist()
MIN_TAGS = 5

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
