#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import dum_bins, galaxy_classes

#%%
results = {
    "random": "data/final/random.csv",
    #"ba": "data/final/ba.csv",
    #"classifier1d": "data/final/filament_galaxies_classifier1d.csv",
    "global": "data/final/global.csv",
    #"global1d": "data/final/global1d.csv",
    #"classifier": "data/final/classifier_0.csv"
    #"kmeans1d": "data/final/kmeans1d.csv",
    "kmeans": "data/final/kmeans.csv",
    "rf": "data/final/rf.csv"
}

dum_values = (dum_bins[:-1] + dum_bins[1:]) / 2
galaxy_classes = galaxy_classes[:2]

#%%
for index, (method, filename) in enumerate(results.items()):
    h = pd.read_csv(filename).values[0:len(galaxy_classes), :]
    plt.plot(range(len(galaxy_classes)), np.sum(h * dum_values, axis=1) / np.sum(h, axis=1), label=method)
    print(method, np.sum(h * dum_values, axis=1) / np.sum(h, axis=1))

plt.legend()

#%%
for index, (method, filename) in enumerate(results.items()):
    h = pd.read_csv(filename).values
    
    #plt.figure(1)
    #plt.plot(range(len(galaxy_classes)), np.sum(h * dum_values, axis=1) / np.sum(h, axis=1), label=method)
    print(method, np.sum(h * dum_values, axis=1) / np.sum(h, axis=1))

    for c in range(len(galaxy_classes)):
        plt.figure(c)
        #plt.ylim((0.009, 0.012))
        plt.title(galaxy_classes[c]["label"])
        plt.plot(dum_values, h[c] / h[c].sum(), label=method)

    #plt.figure(len(sern_bins) + 1 + index)
    #plt.title(method)
    #plt.imshow(hist.values.T, origin="lower")

for c in range(len(galaxy_classes)):
    plt.figure(c)
    plt.gca().legend()