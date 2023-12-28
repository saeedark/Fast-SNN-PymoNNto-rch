from PymoNNto import *
import time
from globparams import *
import scipy as sp

settings = {'dtype': float32, 'synapse_mode': SxD}

def segIndex(inp_list, size):
    import numpy as np
    arr = [[] for _ in range(size)]
    for idx,value in enumerate(inp_list):
        arr[value].append(idx)
    arr = [np.asarray(x) for x in arr]
    out = np.empty(len(arr), dtype=object)
    out[:] = arr
    return out

class LeakyIntegrateAndFire(Behavior):
    def initialize(self, neurons):
        neurons.spikes = neurons.vector('bool')
        neurons.spikesOld = neurons.vector('bool')
        neurons.voltage = neurons.vector()
        self.threshold = self.parameter('threshold')
        self.decay = self.parameter('decay')

    def iteration(self, neurons):
        neurons.spikesOld = neurons.spikes.copy()
        neurons.spikes = neurons.voltage > self.threshold
        #print(np.sum(neurons.spikes))# number of active neurons around 1.5%
        #neurons.voltage.fill(0.0)
        neurons.voltage *= np.invert(neurons.spikes) #reset VR
        neurons.voltage *= self.decay #voltage decay



class Input(Behavior):
    def initialize(self, neurons):
        sparsity = self.parameter('density')
        for s in neurons.synapses(afferent, 'GLU'):
            # s.W = s.matrix('random')
            s.W = sp.sparse.random(*s.matrix_dim(), density=sparsity, dtype=float32)
            # making sure that the order of data remains the same
            s.W = s.W.tocsr()
            s.W = s.W.tocoo()
            s.col_idx = s.W.col
            s.row_idx = s.W.row
            s.pre_idx = segIndex(s.row_idx, s.src.size)
            s.post_idx = segIndex(s.col_idx, s.dst.size)
            s.W = s.W / SIZE
            s.W = s.W.tocsr()
            # s.W /= np.sum(s.W, axis=0) #normalize during initialization

    def iteration(self, neurons):
        neurons.voltage += neurons.vector('random')
        for s in neurons.synapses(afferent, 'GLU'):
            # s.dst.voltage += np.sum(s.W[s.src.spikes], axis=0)
            s.dst.voltage += s.W.dot(s.src.spikes)


class STDP(Behavior):
    def initialize(self, neurons):
        self.speed = self.parameter('speed')

    def iteration(self, neurons):
        for s in neurons.synapses(afferent, 'GLU'):
            # mask = np.ix_(s.src.spikesOld, s.dst.spikes)
            # s.W[mask] += self.speed
            # mask = s.dst.spikes[s.row_idx] * s.src.spikesOld[s.col_idx]
            pre = s.pre_idx[s.src.spikesOld]
            if len(pre):
                pre_mask = np.concatenate(pre)
                post = s.post_idx[s.src.spikes]
                if len(post):
                    post_mask = np.concatenate(post)
                    mask = np.intersect1d(pre_mask, post_mask)
                    data = s.W.data
                    data[mask] += self.speed
                    data[mask] = np.clip(data[mask], 0.0, 1.0)
            


#class Norm(Behavior):
#    def iteration(self, neurons):
#        if neurons.iteration % 10 == 9:
#            for s in neurons.synapses(afferent, 'GLU'):
#                s.W /= np.sum(s.W, axis=0)


net = Network(settings=settings)
NeuronGroup(net, tag='NG', size=SIZE, behavior={
    1: LeakyIntegrateAndFire(threshold=VT, decay=DECAY),
    2: Input(density=DENSITY),
    3: STDP(speed=STDP_SPEED),
    #4: Norm(),
})

if PLOT:
    net.NG.add_behavior(9, EventRecorder('spikes'), False)

SynapseGroup(net, src='NG', dst='NG', tag='GLU')
net.initialize()

if not MEASURE_INDIVIDUALLY:
    start = time.time()
    net.simulate_iterations(DURATION, batch_size=DURATION, measure_block_time=True)
    print("simulation time: ", time.time() - start)
else:
    for i in range(DURATION):
        net.simulate_iteration(measure_behavior_execution_time=True)
    print('module time:', net.time_measures)

if PLOT:
    print('firing rate:', len(net['spikes.i', 0]) / SIZE / DURATION * 1000, 'Hz')
    print(f"Total spikes: {len(net['spikes.i', 0])}")
    plt.plot(net['spikes.t', 0], net['spikes.i', 0], '.k')
    plt.show()
