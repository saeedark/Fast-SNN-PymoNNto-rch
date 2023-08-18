import matplotlib.pyplot as plt
import numpy as np
import csv

fig, ax = plt.subplots(1, 3)

fig.set_figwidth(12)
fig.set_figheight(4)

ekw = dict(ecolor=(0, 0, 0, 1.0), lw=1, capsize=3, capthick=1)

color1 = (0, 176/255, 80/255, 1)
color2 = (1/255, 127/255, 157/255, 1)
color3 = (253/255, 97/255, 0, 1)

color11 = (0, 176/255, 80/255, 0.7)

lookup = {'brian_LIF.py': ['Brian2', color2],
          'brian_LIF_cpp.py': ['Brian2 C++', color2],
          'brian_LIF_gpu.py': ['Brian2 Cuda', color2],
          'nest_native_LIF.py': ['Nest', color3],
          'pymonnto_fast_LIF.py': ['PymoNNto', color11],
          'pymonnto_slow_LIF.py': ['PymoNNto', color11],
          'pymonntorch_fast_LIF_cpu.py': ['Pymonntorch CPU', color1],
          'pymonntorch_fast_LIF_cuda.py': ['Pymonntorch GPU', color1],
          'pymonntorch_slow_LIF_cpu.py': ['Pymonntorch CPU naive', color1],
          'pymonntorch_slow_LIF_cuda.py': ['Pymonntorch GPU naive', color1],

          'brian_izh.py': ['Brian2', color2],
          'brian_izh_cpp.py': ['Brian2 C++', color2],
          'brian_izh_cuda.py': ['Brian2 Cuda', color2],
          'pymonnto_izh.py': ['PymoNNto', color11],
          'pymonntorch_izh_cpu.py': ['PymoNNtorch CPU', color1],
          'pymonntorch_izh_cuda.py': ['PymoNNtorch GPU', color1],
          'pynn_nest_izh.py': ['Nest (PyNN)', color1]
          }

def load(filename):
    #return np.random.rand(3,7)+3
    #return np.array([[10, 15, 10, 300, 310, 500, 550], [2, 3, 8, 345, 333, 546, 522], [10, 15, 10, 300, 310, 500, 550]])

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


sim_col, data = load('test.csv')
measurements = np.mean(data, axis=0)
err = np.std(data, axis=0)
simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
index = list(range(len(sim_col)))


text_gap = np.max(measurements)*0.01
for i, s, m, e, c in zip(index, simulators, measurements, err, colors):
    ax[0].bar(i, m, width=0.8, color=c, yerr=e, error_kw=ekw)
    ax[0].text(i, m+e+text_gap, '{0:.2f}'.format(m)+'s', ha='center', va='bottom', color=c)  # , color='gray'

ax[0].tick_params(axis='both', which='both', length=0)
ax[0].set_yticks([], [])
ax[0].set_ylim([0, np.max(measurements)*1.3])
ax[0].set_xticks(np.array(index), simulators, rotation=30, ha="right")
ax[0].spines[['left', 'right', 'top']].set_visible(False)
ax[0].set_title('Simple LIF with\nOne Step STDP', fontweight='bold')

for xtick, color in zip(ax[0].get_xticklabels(), colors):
    xtick.set_color(color)

#ax[0].set_ylabel('t')






sim_col, data = load('test.csv')
measurements = np.mean(data, axis=0)
err = np.std(data, axis=0)
simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
index = list(range(len(sim_col)))

text_gap = np.max(measurements)*0.01
for i, s, m, e, c in zip(index, simulators, measurements, err, colors):
    ax[1].bar(i, m, width=0.8, color=c, yerr=e, error_kw=ekw)
    ax[1].text(i, m + e+text_gap, '{0:.2f}'.format(m)+'s', ha='center', va='bottom', color=c)


ax[1].tick_params(axis='both', which='both', length=0)
ax[1].set_yticks([], [])
ax[1].set_ylim([0, np.max(measurements)*1.3])
ax[1].set_xticks(index, simulators, rotation=30, ha="right")
ax[1].spines[['left', 'right', 'top']].set_visible(False)
ax[1].set_title('Izhikevich with\nTrace STDP', fontweight='bold')

for xtick, color in zip(ax[1].get_xticklabels(), colors):
    xtick.set_color(color)








sim_col, data = load('test.csv')
measurements = np.mean(data, axis=0)
err = np.std(data, axis=0)
simulators = [s for s, c in sim_col]
colors = [c for s, c in sim_col]
index = list(range(len(sim_col)))

text_gap = np.max(measurements)*0.01
for i, s, m, e, c in zip(index, simulators, measurements, err, colors):
    ax[2].bar(i, m, width=0.8, color=c, yerr=e, error_kw=ekw)
    ax[2].text(i, m + e+text_gap, '{0:.2f}'.format(m)+'s', ha='center', va='bottom', color=c)


ax[2].tick_params(axis='both', which='both', length=0)
ax[2].set_yticks([], [])
ax[2].set_ylim([0, np.max(measurements)*1.3])
ax[2].set_xticks(index, simulators, rotation=30, ha="right")
ax[2].spines[['left', 'right', 'top']].set_visible(False)
ax[2].set_title('Simulator\nstartup time', fontweight='bold')

for xtick, color in zip(ax[2].get_xticklabels(), colors):
    xtick.set_color(color)


fig.tight_layout()
plt.show()








#ax.barh(y+0.2, c/np.max(c)*np.max(n), height=0.4, color=color1)

#ax2.scatter([np.max(c)], [0])
#ax.set_xlim([0, np.max(n)])
#ax2.set_xlim([0, np.max(c)])


#ax.set_ylabel("spike possibility")
#ax.set_xlabel("voltage")
#ax[0].set_yticks([])
#ax.set_yticks(y, sorted_alphabet)
#ax2.set_yticks([])

#ax2.xaxis.tick_top()
#ax2.xaxis.set_label_position('top')

#ax.set_xlabel('number of neurons clustered by weights', color=colorE)  # we already handled the x-label with ax1
#ax.tick_params(axis='x', labelcolor=colorE)

#ax2.set_xlabel('number of char in input', color=color1)  # we already handled the x-label with ax1
#ax2.tick_params(axis='x', labelcolor=color1)

##ax.spines[['top', 'bottom']].set_visible(False)
##ax2.spines[['top', 'bottom']].set_visible(False)

#ax2.spines[['left', 'right', 'left']].set_visible(False)

#plt.text(x=-0.4, y=1.1, s='A', size=20, weight='bold')