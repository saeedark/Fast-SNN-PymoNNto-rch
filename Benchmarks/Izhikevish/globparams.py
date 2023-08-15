DURATION = 300
SIZE = 2500

A, B, C, D = 0.02, 0.04, -65.0, 2.0
THRESHOLD = 30.0
V_MEAN, U_MEAN = -65, 12
V_STD, U_STD= 7, 7

TRACE_TAU = 20.0
TRACE_TAIL = 0.01

DIRAC_STRENGTH = 1 / SIZE
A_PLUS, A_MINUS = 0.01, 0.012
W_MIN, W_MAX = 0.0, 1.0

OFFSET = 15
NOISE_MEAN = 0
NOISE_STD = 1


SPIKE_RECORD = False

REPORT_FUNC = '''
    if (completed == 1.0) std::cout << "simulation time: " << elapsed << std::endl << std::flush;
'''
