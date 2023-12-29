from pymonntorch import *
import time
import torch
from matplotlib import pyplot as plt
import numpy as np
from globparams import *
import random

settings = {'dtype': torch.float32, 'synapse_mode': "SxD", 'device': 'cpu'}


class LeakyIntegrateAndFire(Behavior):
    def initialize(self, neurons):
        neurons.spikes = neurons.vector(dtype=torch.bool)
        neurons.spikesOld = neurons.vector(dtype=torch.bool)
        neurons.voltage = neurons.vector()
        self.threshold = self.parameter('threshold')
        self.decay = self.parameter('decay')

    def forward(self, neurons):
        neurons.spikesOld = neurons.spikes.clone()
        neurons.spikes = neurons.voltage > self.threshold
        #print(np.sum(neurons.spikes))# number of active neurons around 1.5%
        #neurons.voltage.fill(0.0)
        neurons.voltage *= ~neurons.spikes #reset
        neurons.voltage *= self.decay #voltage decay



class Input(Behavior):
    def initialize(self, neurons):
        sparsity = self.parameter('density')
        for s in neurons.synapses('afferent', 'GLU'):
            # s.W = s.matrix('random')
            n_row, n_col = s.matrix_dim()
            nnz = int(n_row * n_col * sparsity)
            both_indices =  torch.tensor(random.sample(range(n_row*n_col), nnz), device=s.device)
            s.row_idx = both_indices % n_col
            s.col_idx = both_indices // n_col
            indices = torch.stack([s.col_idx, s.row_idx])
            values =  s.tensor(mode='uniform', dim=(nnz,))
            s.W = torch.sparse_coo_tensor(indices, values, s.matrix_dim())
            s.W = s.W.coalesce()
            s.W = s.W / SIZE
            s.W = s.W.to_sparse_csc()
            # s.W /= torch.sum(s.W, axis=0) #normalize during initialization

    def forward(self, neurons):
        neurons.voltage += neurons.vector('random')
        for s in neurons.synapses('afferent', 'GLU'):
            # s.dst.voltage += torch.sum(s.W[s.src.spikes], axis=0)
            # s.dst.voltage += torch.mv(s.W, s.src.spikes*1.0)
            # s.dst.voltage += torch.mm(s.src.spikes.view(1,-1)*1.0, s.W).view(-1,)
            s.dst.voltage += torch.matmul(s.src.spikes*1.0, s.W)
            # pass


class STDP(Behavior):
    def initialize(self, neurons):
        self.speed = self.parameter('speed')

    def forward(self, neurons):
        for s in neurons.synapses('afferent', 'GLU'):
            # mask = (torch.where(s.src.spikesOld)[0].view(-1, 1), torch.where(s.dst.spikes)[0].view(1, -1))
            # mask = torch.gather(s.dst.spikes,0,s.row_idx) * torch.gather(s.src.spikesOld,0,s.col_idx)
            mask = s.dst.spikes[s.row_idx] * s.src.spikesOld[s.col_idx]
            data = s.W.values()[:]
            data[mask] += self.speed
            data[mask] = torch.clip(data[mask], 0.0, 1.0)
            


#class Norm(Behavior):
#    def iteration(self, neurons):
#        if neurons.iteration % 10 == 9:
#            for s in neurons.synapses(afferent, 'GLU'):
#                s.W /= np.sum(s.W, axis=0)


net = Network(**settings)
NeuronGroup(net=net, tag='NG', size=SIZE, behavior={
    1: LeakyIntegrateAndFire(threshold=VT, decay=DECAY),
    2: Input(density=DENSITY),
    3: STDP(speed=STDP_SPEED),
    #4: Norm(),
    #5: EventRecorder(['spikes'])
})

if PLOT:
    net.NG.add_behavior(9, EventRecorder('spikes'), False)

SynapseGroup(net=net, src='NG', dst='NG', tag='GLU')
net.initialize()

start = time.time()
net.simulate_iterations(DURATION)
print("simulation time: ", time.time()-start)

if PLOT:
    plt.plot(net['spikes.t', 0].cpu(), net['spikes.i', 0].cpu(), '.k')
    plt.show()
