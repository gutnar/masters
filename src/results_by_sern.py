#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
results = {
    "random": "data/final/filament_galaxies_random.csv",
    #"ba": "data/final/filament_galaxies_ba.csv",
    #"global1d": "data/final/filament_galaxies_global1d.csv",
    #"classifier1d": "data/final/filament_galaxies_classifier1d.csv",
    "global": "data/final/global_0.csv",
    "classifier": "data/final/classifier_0.csv"
}

dum_values = (np.linspace(0, 1, 101)[:-1] + np.linspace(0, 1, 101)[1:]) / 2
sern_values = (np.linspace(0, 20, 51)[:-1] + np.linspace(0, 20, 51)[1:]) / 2
sern_bins = [0, 2, 20]

#%%
for index, (method, filename) in enumerate(results.items()):
    hist = pd.read_csv(filename)

    h = np.zeros((len(sern_bins) - 1, 100))
    for i in range(len(sern_bins) - 1):
        h[i] = hist[(sern_values >= sern_bins[i]) & (sern_values < sern_bins[i + 1])].sum(axis=0)
    
    plt.figure(1)
    plt.plot(sern_bins[1:], np.sum(h * dum_values, axis=1) / np.sum(h, axis=1), label=method)
    print(np.sum(h * dum_values, axis=1) / np.sum(h, axis=1))

    for i in range(len(sern_bins) - 1):
        plt.figure(2 + i)
        plt.title("%d < sern < %d" % (sern_bins[i], sern_bins[i+1]))
        plt.plot(dum_values, h[i] / h[i].sum(), label=method)

    #plt.figure(len(sern_bins) + 1 + index)
    #plt.title(method)
    #plt.imshow(hist.values.T, origin="lower")

plt.figure(1)
plt.gca().legend()

for i in range(len(sern_bins) - 1):
    plt.figure(2 + i)
    plt.gca().legend()
