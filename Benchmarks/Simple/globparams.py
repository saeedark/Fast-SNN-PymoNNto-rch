import sys
PLOT = not 'no_plot' in sys.argv
MEASURE_INDIVIDUALLY = False

DURATION = 300
SIZE = 2500

VT = 6.0
VR = 0.0
STDP_SPEED = 0.001
DECAY = 0.9
OM_DECAY = 1 - DECAY

REPORT_FUNC = '''
    if (completed == 1.0) std::cout << "simulation time: " << elapsed << std::endl << std::flush;
'''
