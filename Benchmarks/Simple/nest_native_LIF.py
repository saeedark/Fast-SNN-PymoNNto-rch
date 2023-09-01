import nest
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from globparams import *

from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils

#############################################################
#model generation
#############################################################

simple_neuron_str = """
neuron simple_neuron:

    state:
        v mV = 0 mV

    equations:
        v' = (I_e/pA * mV - v * decay) / ms

    parameters:
        vt real = 6.1
        vr real = 0.0
        input_strength real = 1.0
        decay real = 0.1


    input:
        spikes mV <- spike
        I_e pA <- continuous

    output:
        spike

    update:
        integrate_odes()
        v += spikes*input_strength

        # threshold crossing
        if v >= vt * mV:
            v = vr * mV
            emit_spike()
"""

simple_stdp_synapse = """
synapse stdp_nn_symm:
    state:
        w real = 1.
        pre_trace real = 0.

    parameters:
        d ms = 1 ms  @nest::delay
        tau_tr_pre ms = 1 ms
        stdp_speed real = 0.01

    equations:
        pre_trace' = -pre_trace / tau_tr_pre

    input:
        pre_spikes real <- spike
        post_spikes real <- spike

    output:
        spike

    onReceive(post_spikes):
        if pre_trace>0:
            w += stdp_speed * pre_trace

    onReceive(pre_spikes):
        pre_trace = 1
        w = w
        deliver_spike(w, d)
"""

module_name, neuron_model_name, synapse_model_name = NESTCodeGeneratorUtils.generate_code_for(
    simple_neuron_str,
    simple_stdp_synapse,
    post_ports=["post_spikes"])

nest.Install(module_name)


#############################################################
#simulation
#############################################################


# Set up the NEST simulation
nest.ResetKernel()
nest.SetKernelStatus({'resolution': 1.0, 'print_time': False})

# Define parameters
num_neurons = SIZE
simulation_time = DURATION  # ms
dt = 1.0  # ms

# Create neurons
neuron_params = {'vt': VT,
                 'vr': VR,
                 'decay': OM_DECAY}
neurons = nest.Create(neuron_model_name, num_neurons, params=neuron_params)

# Create synapses
synapse_params = {'synapse_model': synapse_model_name,
                  'w': nest.random.uniform(min=0.0, max=1.0/num_neurons),
                  'stdp_speed': STDP_SPEED}
nest.Connect(neurons, neurons, 'all_to_all', synapse_params)


#add voltage fluctuations to neurons
for i in range(num_neurons):
    times = list(np.arange(1.0, 100.0, 1.0))
    values = list(np.random.rand(int(simulation_time)-1))
    ng = nest.Create('step_current_generator')
    ng.set({"amplitude_times": times, "amplitude_values": values})
    nest.Connect(ng, neurons[i])

#random noise
#times = list(np.arange(1.0, 100.0, 1.0))
#values = list(np.random.rand(99))
#ng = nest.Create('step_current_generator', num_neurons)
#ng.set({"amplitude_times": times, "amplitude_values": nest.random.uniform(min=0.0, max=1.0)})
#nest.Connect(ng, neurons, "one_to_one")

if RECORD:
    sr = nest.Create("spike_recorder")
    nest.Connect(neurons, sr)


#print(f"Start time: {time.time()}")
#start = time.time()
nest.Simulate(1.0)
#print(time.time()-start)
#print(f"End time: {time.time()}")

start = time.time()
nest.Simulate(simulation_time-1)
print("simulation time: ", time.time() - start)

if RECORD:
    spike_rec=nest.GetStatus(sr, keys='events')[0]
    print(f"Total spikes: {len(spike_rec['times'])}")

if PLOT:
    spike_rec=nest.GetStatus(sr, keys='events')[0]
    plt.plot(spike_rec['times'], spike_rec['senders'], '.k')
    plt.ylabel("neurons")
    plt.xlabel("t")
    plt.show()
