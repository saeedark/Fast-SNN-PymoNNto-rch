from globparams import *
from ANNarchy import *
import os

setup(dt=1.0, paradigm="openmp", precision="float", num_threads=os.cpu_count())

custom_synapse = Synapse(
    parameters=f"""
    speed = {STDP_SPEED}
    """,
    equations="",
    pre_spike="g_target += w",
    post_spike="""
    w = ite((t - t_pre) < 2, ite((t-t_pre >= 1),clip(w + speed, 0.0, 1.0),w), w)
    """,
)

# Definition of the neuron
LIF = Neuron(
    parameters=f"""
        tau = 1.0
        vt = {VT}
        vr = {VR} 
        omdc = {OM_DECAY}
    """,
    equations="""
        tau * dv/dt = g_exc + Uniform(0.0, 1.0) - v * omdc
        tau * dg_exc/dt = - g_exc
    """,  #
    spike="""
        v > vt
    """,
    reset="""
        v = vr
    """,
)

# Output neuron
Output = Population(name="Output", geometry=SIZE, neuron=LIF)

# Projection learned using STDP
proj = Projection(pre=Output, post=Output, target="exc", synapse=custom_synapse)
proj.connect_all_to_all(weights=Uniform(0.0, 1 / SIZE))


# Compile the network
compile()

if PLOT:
    Mo = Monitor(Output, "spike")

# Start the simulation
print("Start the simulation")
start = time.time()
simulate(DURATION, measure_time=True)
print("simulation time: ", time.time() - start)


if PLOT:
    # Retrieve the recordings
    # input_spikes = Mi.get('spike')
    output_spikes = Mo.get("spike")

    # print(output_spikes)

    # Compute the mean firing rates during the simulation
    # print('Mean firing rate in the input population: ' + str(Mi.mean_fr(input_spikes)) )
    # print('Mean firing rate of the output neuron: ' + str(Mo.mean_fr(output_spikes)) )

    # Compute the instantaneous firing rate of the output neuron
    output_rate = Mo.smoothed_rate(output_spikes, 100.0)

    # Receptive field after simulation
    # weights = proj.w[0]

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
