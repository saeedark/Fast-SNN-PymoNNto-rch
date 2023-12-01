from globparams import *
from ANNarchy import *
import numpy as np
import time
import os

setup(dt=1.0, paradigm="cuda", precision="float", num_threads=os.cpu_count())

# Definition of the neuron
Izhikevich = Neuron(
    parameters=f"""
        a = {A}
        b = {B}
        c = {C}
        d = {D} 
        v_thresh = {THRESHOLD}
    """,
    equations=f"""
        dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I + Normal({NOISE_MEAN}, {NOISE_STD}) 
        du/dt = a * (b*v - u) 
        dI/dt = (-I + {OFFSET})
    """,
    spike="""
        v >= v_thresh
    """,
    reset="""
        v = c
        u += d
    """,
)

Output = Population(name="Output", geometry=SIZE, neuron=Izhikevich)

Output.v = np.random.normal(V_MEAN, V_STD, size=SIZE)
Output.u = np.random.normal(U_MEAN, U_STD, size=SIZE)

CustomSTDP = Synapse(
    parameters=f"""
        tau_pre = {TRACE_TAU} : projection
        tau_post = {TRACE_TAU} : projection
        cApre = {A_MINUS} : projection
        cApost = {A_PLUS} : projection
        wmax = 1.0 : projection
    """,
    equations="""
        tau_pre * dApre/dt = - Apre 
        tau_post * dApost/dt = - Apost 
    """,
    pre_spike=f"""
        v_target += (w * {DIRAC_STRENGTH})
        Apre += cApre
        w = clip(w - (Apre * (w - 0.0)), 0.0 , wmax)
    """,
    post_spike="""
        Apost += cApost
        w = clip(w + (Apost * (wmax - w)), 0.0 , wmax)
    """,
)

# Projection learned using STDP
proj = Projection(
    pre=Output,
    post=Output,
    target="I",
    synapse=CustomSTDP,
)
proj.connect_all_to_all(weights=Uniform(0.0, 1.0))


# Compile the network
compile()

if PLOT:
    Mo = Monitor(Output, "spike")

# Start the simulation
# print('Start the simulation')
start = time.time()
simulate(DURATION)
print("simulation time: ", time.time() - start)

if PLOT:
    output_spikes = Mo.get("spike")

    # Compute the instantaneous firing rate of the output neuron
    output_rate = Mo.smoothed_rate(output_spikes, 100.0)

    import matplotlib.pyplot as plt

    for k, v in output_spikes.items():
        plt.plot(np.array(v), np.array(v) * 0 + k, ".k", c="black")
        # for w in v:
        #    plt.scatter([k], [w])

    # plt.figure(figsize=(20, 15))
    # plt.subplot(3,1,1)
    # plt.plot(output_rate[0, :])
    # plt.subplot(3,1,2)
    # plt.plot(weights, '.')
    # plt.subplot(3,1,3)
    # plt.hist(weights, bins=20)
    plt.show()
