import matplotlib.pyplot as plt
import numpy as np
import csv

def load2(filename):
    sim = []
    col = []
    num = []
    mes = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i>0:
                sim.append(lookup[row[1]][0])
                col.append(lookup[row[1]][1])
                num.append(int(row[2]))
                mes.append(float(row[3]))
    return np.array(sim), np.array(col), np.array(num), np.array(mes)

ekw = dict(ecolor=(0, 0, 0, 1.0), lw=1, capsize=3, capthick=1)

color1 = (0, 176/255, 80/255, 1)
color2 = (1/255, 127/255, 157/255, 1)
color3 = (120/255,110/255,120/255,1)#(117/255,117/255,117/255,1)#(253/255, 97/255, 0, 1)
color4 = (100/255, 0, 124/255, 0.8) 
#color11 = (0, 176/255, 80/255, 0.7)
color11 = (0, 176/255*0.5, 80/255*0.5, 0.7)
#color11 = (0, 0, 0, 1.0)

lookup = {'brian_LIF.py': ['Brian 2', color2],
          'brian_LIF_cpp.py': ['Brian 2 C++', color2],
          'brian_LIF_gpu.py': ['Brian 2 GPU', color2],
          'nest_native_LIF.py': ['NEST', color3],
          'pymonnto_fast_LIF.py': ['PymoNNto', color11],
          'pymonnto_slow_LIF.py': ['PymoNNto', color11],# naive
          'pymonntorch_fast_LIF_cpu.py': ['PymoNNtorch CPU', color1],
          'pymonntorch_fast_LIF_cuda.py': ['PymoNNtorch GPU', color1],
          'pymonntorch_slow_LIF_cpu.py': ['PymoNNtorch CPU', color1],# naive
          'pymonntorch_slow_LIF_cuda.py': ['PymoNNtorch GPU', color1],# naive

          'brian_izh.py': ['Brian 2', color2],
          'brian_izh_cpp.py': ['Brian 2 C++', color2],
          'brian_izh_cuda.py': ['Brian 2 GPU', color2],
          'pymonnto_izh.py': ['PymoNNto', color11],
          'pymonntorch_izh_cpu.py': ['PymoNNtorch CPU', color1],
          'pymonntorch_izh_cuda.py': ['PymoNNtorch GPU', color1],
          'pynn_nest_izh.py': ['NEST (PyNN)', color3],
          'annarchy_izh_cpu.py': ['ANNarchy CPU', color4],
          'annarchy_LIF_cpu.py': ['ANNarchy CPU', color4],
          'annarchy_izh_cuda.py': ['ANNarchy GPU', color4],
          'annarchy_LIF_cuda.py': ['ANNarchy GPU', color4],
          }

markers_lookup = {"⋅⋅⋅ Pymonntorch GPU": ':',
"⋅⋅⋅ PymoNNtorch GPU": ':',
		  "-- Pymonntorch CPU": '--',
		  "-- Pymonntorch CPU": '--',
		  "-- PymoNNtorch CPU": '--',
		  "-- PymoNNtorch CPU": '--',
		  "— PymoNNto": '-',
"⋅⋅⋅ ANNarchy GPU": ':',
"-- ANNarchy CPU": '--',
"— NEST": '-',
"⋅⋅⋅ Brian 2 GPU": ":",
"-- Brian 2 C++": "--",
"— Brian 2": "-",		
'— NEST (PyNN)': "-",  
}

def load(filename):
    sim_col = []
    measurements = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i==0:
                sim_col = [lookup[s] for s in row[1:]]
            else:
                measurements.append([float(s) for s in row])

    return sim_col, np.array(measurements)[:,1:] #remove enumeration in first column




lookup = {'brian_LIF.py': [ '— Brian 2', color2],
          'brian_LIF_cpp.py': ['-- Brian 2 C++', color2],
          'brian_LIF_gpu.py': ['⋅⋅⋅ Brian 2 GPU', color2],
          'nest_native_LIF.py': ['— NEST', color3],
          'pymonnto_fast_LIF.py': ['— PymoNNto', color11],
          'pymonnto_slow_LIF.py': ['— PymoNNto', color11],# naive
          'pymonntorch_fast_LIF_cpu.py': ['-- PymoNNtorch CPU', color1],
          'pymonntorch_fast_LIF_cuda.py': ['⋅⋅⋅ PymoNNtorch GPU', color1],
          'pymonntorch_slow_LIF_cpu.py': ['-- PymoNNtorch CPU', color1],# naive
          'pymonntorch_slow_LIF_cuda.py': ['⋅⋅⋅ PymoNNtorch GPU', color1],# naive

          'annarchy_izh_cpu.py': ['-- ANNarchy CPU', color4],
          'annarchy_LIF_cpu.py': ['-- ANNarchy CPU', color4],
          'annarchy_izh_cuda.py': ['⋅⋅⋅ ANNarchy GPU', color4],
          'annarchy_LIF_cuda.py': ['⋅⋅⋅ ANNarchy GPU', color4],
          'brian_izh.py': ['— Brian 2', color2],
          'brian_izh_cpp.py': ['-- Brian 2 C++', color2],
          'brian_izh_cuda.py': ['⋅⋅⋅ Brian 2 GPU', color2],
          'pymonnto_izh.py': ['— PymoNNto', color11],
          'pymonntorch_izh_cpu.py': ['-- PymoNNtorch CPU', color1],
          'pymonntorch_izh_cuda.py': ['⋅⋅⋅ PymoNNtorch GPU', color1],
          'pynn_nest_izh.py': ['— NEST (PyNN)', color3]
          }


#1
###############################################################################
#2

fig, ax = plt.subplots(1, 3, sharey=True)
fig.set_figwidth(12)
fig.set_figheight(5)



fig.suptitle('LIF with Step STDP')


axis = ax[0]
sim, col, num, mes = load2('Results/AWS/low/LIF.csv')
for s in np.flip(np.unique(sim)):
    idx = sim==s
    x = np.sort(np.unique(num[idx]))
    y = np.array([np.mean(mes[idx][num[idx]==n]) for n in x])
    e = np.array([np.std(mes[idx][num[idx]==n]) for n in x])
    axis.fill_between(x, y - e, y + e, alpha=0.5, edgecolor=col[idx][0], facecolor=col[idx][0], linewidth=0) #, c=col[idx][0]
    # axis.plot(x, y, c=col[idx][0])
    axis.plot(x, y, c=col[idx][0], linestyle=markers_lookup[s], label=s)
axis.semilogy()
axis.tick_params(axis='x', which='both', length=3)
# axis.tick_params(axis='y', which='both', length=0)
# axis.set_yticks([], [])
axis.set_ylabel('Simulation Time (log scale)')

axis.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000], [0, 2500, 5000, 7500, 10000, 12500, 15000])
axis.spines[['right', 'top']].set_visible(False)
axis.set_xlim([0, 15000])
# axis.set_title('Number of Neurons', x=0.87, y=0, pad=-14, fontsize=10)
# fig.legend(handles, labels, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0.01), prop={'size':8} )



axis = ax[1]
sim, col, num, mes = load2('Results/AWS/LIF.csv')
for s in np.flip(np.unique(sim)):
    idx = sim==s
    x = np.sort(np.unique(num[idx]))
    y = np.array([np.mean(mes[idx][num[idx]==n]) for n in x])
    e = np.array([np.std(mes[idx][num[idx]==n]) for n in x])
    axis.fill_between(x, y - e, y + e, alpha=0.5, edgecolor=col[idx][0], facecolor=col[idx][0], linewidth=0) #, c=col[idx][0]
    # axis.plot(x, y, c=col[idx][0])
    axis.plot(x, y, c=col[idx][0], linestyle=markers_lookup[s], label=s)
axis.semilogy()
axis.tick_params(axis='x', which='both', length=3)
# axis.tick_params(axis='y', which='both', length=0)
# axis.set_yticks([], [])

axis.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000], [0, 2500, 5000, 7500, 10000, 12500, 15000])
axis.spines[['right', 'top']].set_visible(False)
axis.set_xlim([0, 15000])
# axis.set_title('Number of Neurons', x=0.87, y=0, pad=-14, fontsize=10)



axis = ax[2]
sim, col, num, mes = load2('Results/AWS/high/LIF.csv')
for s in np.flip(np.unique(sim)):
    idx = sim==s
    x = np.sort(np.unique(num[idx]))
    y = np.array([np.mean(mes[idx][num[idx]==n]) for n in x])
    e = np.array([np.std(mes[idx][num[idx]==n]) for n in x])
    axis.fill_between(x, y - e, y + e, alpha=0.5, edgecolor=col[idx][0], facecolor=col[idx][0], linewidth=0) #, c=col[idx][0]
    # axis.plot(x, y, c=col[idx][0])
    axis.plot(x, y, c=col[idx][0], linestyle=markers_lookup[s], label=s)
axis.semilogy()
axis.tick_params(axis='x', which='both', length=3)
# axis.tick_params(axis='y', which='both', length=0)
# axis.set_yticks([], [])

axis.set_xticks([0, 2500, 5000, 7500, 10000], [0, 2500, 5000, 7500, 10000])
axis.spines[['right', 'top']].set_visible(False)
axis.set_xlim([0, 10000])
axis.set_title('Number of Neurons', x=0.87, y=0, pad=-14, fontsize=10)
# axis.set_xlabel("Number of Neurons", loc='right')



# ax[0].text(x=0, y=160, s='C', size=17)
# ax[1].text(x=0, y=7.5, s='D', size=17)

# ax[0].set_title("A", loc='left')
# ax[1].set_title("B", loc='left')
# ax[2].set_title("C", loc='left')

ax[0].set_title('Low Spike Rate', x=0.47, y=0.95, pad=0, fontsize=10)
ax[1].set_title('Normal Spike Rate', x=0.47, y=0.95, pad=0, fontsize=10)
ax[2].set_title('High Spike Rate', x=0.47, y=0.95, pad=0, fontsize=10)


axis.text(5900, 0.0013, "Number of Neurons")

#ax[0].text(x=100, y=0, s=' ', size=20, weight='bold')

#ax[0].set_ylabel('compute time')

fig.tight_layout()
plt.savefig('lif_rate.jpg', dpi=600)
plt.show()


#2
###############################################################################

