
# Pranav Minasandra
# pminasandra.github.io
# March 23, 2026

import os
import os.path


#Directories
PROJECTROOT = open(".cw", "r").read().rstrip()
DATA = os.path.join(PROJECTROOT, "Data")
FIGURES = os.path.join(PROJECTROOT, "Figures")

formats=['png', 'pdf', 'svg']

#Miscellaneous
SUPPRESS_INFORMATIVE_PRINT = False
