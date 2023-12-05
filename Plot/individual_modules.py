import matplotlib.pyplot as plt

#overhead test: 0.001092708

labels = []
values = []
#{1: 5.002260208129883, 2: 20.522356033325195, 3: 15.061616897583008} #100 iterations
pymonnto_fast_LIF = {'Overhead':0.001092708, 'Input': 20.522356033325195/300, 'LeakyIntegrate-\nAndFire': 5.002260208129883/300, 'STDP': 15.061616897583008/300}
for k,v in pymonnto_fast_LIF.items():
    values.append(v)
    labels.append(k+' (~{0:.0f}μs)'.format(v*1000)) #/iterations*(ms to μs)

fig, ax = plt.subplots()
plt.pie(values, labels=labels)
plt.savefig('LIF_Modules.jpg', dpi=600)
plt.show()

labels = []
values = []
#{1: 74.15080070495605, 2: 15.035152435302734, 5: 882.4601173400879} #300 iterations
pymonnto_fast_IZH = {'Overhead': 0.001092708, 'Input': 74.15080070495605/300, 'Izhikevich': 15.035152435302734/300, 'STDP': 882.4601173400879/300}
for k,v in pymonnto_fast_IZH.items():
    values.append(v)
    labels.append(k+' (~{0:.0f}μs)'.format(v*1000))

fig, ax = plt.subplots()
plt.pie(values, labels=labels)
plt.savefig('Izh_Modules.jpg', dpi=600)
plt.show()
