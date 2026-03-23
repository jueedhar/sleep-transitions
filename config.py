
# Pranav Minasandra
# pminasandra.github.io
# March 23, 2026

import os
import os.path


#Directories
PROJECTROOT = open(".cw", "r").read().rstrip()
DATA = os.path.join(PROJECTROOT, "Data")
FIGURES = os.path.join(PROJECTROOT, "Figures")

MASTER_DATA_SHEET = os.path.join(config.DATA, "baboon_sleep_wake_transitions.parquet")

formats=['png', 'pdf', 'svg']

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
