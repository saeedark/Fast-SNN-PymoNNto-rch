from brian2 import *
import time

defaultclock.dt = 1*ms
prefs.core.default_float_dtype = float32

vt = 6.1
vr = 0.0
input_strength = 1.0
stdp_speed = 0.001
decay = 1 - 0.9


eqs_neurons = '''
dv/dt = (ge + rand() - v*decay) / (1*ms) : 1
dge/dt = -ge / (1*ms) : 1
dspiked/dt = -spiked / (1*ms) : 1
'''

N = NeuronGroup(15000, eqs_neurons, threshold='v>vt', reset='v = vr', method='euler')

synaptic_model = '''
w : 1
'''

pre = '''
ge_post += w * input_strength
spiked_pre = 1
'''

post = '''
w = clip(w + spiked_pre * stdp_speed, 0.0, 1.0) 
'''

S = Synapses(N, N, synaptic_model, on_pre=pre, on_post=post)

S.connect()
S.w = 'rand()' #initialize
S.w /= sum(S.w, axis=0) #normalize


M = SpikeMonitor(N)


start = time.time()
run(100*ms, report='text')
print("simulation time: ", time.time()-start)


plot(M.t/ms, M.i, '.')
show()



#@network_operation(when='start', dt=10*ms)#, dt=10*ms
#def syn_norm():
#    print('test')
#    S2.w /= sum(S2.w, axis=0)

