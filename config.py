
# Pranav Minasandra
# pminasandra.github.io
# March 23, 2026

import os
import os.path


#Directories
PROJECTROOT = open(".cw", "r").read().rstrip()
DATA = os.path.join(PROJECTROOT, "Data")
FIGURES = os.path.join(PROJECTROOT, "Figures")

MASTER_DATA_SHEET = os.path.join(DATA, "baboon_sleep_wake_transitions.parquet")

formats=['png', 'pdf', 'svg']

# Simulation details

PERCENTILE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
MIN_TAGS = 5

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
