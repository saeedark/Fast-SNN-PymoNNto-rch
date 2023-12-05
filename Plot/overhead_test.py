from PymoNNto import *
import time

settings = {'dtype': float32, 'synapse_mode': SxD}

class SpikeGeneration(Behavior):
    def initialize(self, neurons):
        return True

    def iteration(self, neurons):
        return True

class Input(Behavior):
    def initialize(self, neurons):
        return True

    def iteration(self, neurons):
        return True


class STDP(Behavior):
    def initialize(self, neurons):
        return True

    def iteration(self, neurons):
        return True

net = Network(settings=settings)
NeuronGroup(net, tag='NG', size=2500, behavior={
    1: SpikeGeneration(),
    2: Input(),
    3: STDP()
})

SynapseGroup(net, src='NG', dst='NG', tag='GLU')
net.initialize()

#time.time_ns()

start = time.time_ns()
net.simulate_iterations(100000)
time_ns=(time.time_ns()-start)/100000

time_mus=time_ns/1000
time_ms=time_mus/1000
time_s=time_ms/1000
print("simulation time: ", time_ms, "ms")


