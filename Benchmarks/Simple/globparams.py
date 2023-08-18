import sys
PLOT = not 'no_plot' in sys.argv

DURATION = 100
SIZE = 10000

VT = 6.1
VR = 0.0
STDP_SPEED = 0.001
DECAY = 0.9
OM_DECAY = 1 - DECAY