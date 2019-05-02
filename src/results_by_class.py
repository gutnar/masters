#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import dum_bins, galaxy_classes

#%%
plt.rcParams['figure.figsize'] = [25, 16]
plt.rcParams["font.size"] = 16

results = {
    "random": ["data/final/random.csv", 2],
    "ba": ["data/final/ba.csv", 10],
    #"classifier1d": "data/final/filament_galaxies_classifier1d.csv",
    "global": ["data/final/global.csv", 2],
    #"global1d": "data/final/global1d.csv",
    #"kmeans1d": "data/final/kmeans1d.csv",
    #"kmeans": "data/final/kmeans.csv",
    "rf": ["data/final/rf.csv", 2]
}

dum_values = (dum_bins[:-1] + dum_bins[1:]) / 2
#galaxy_classes = galaxy_classes[:2]

#%%
for index, (method, method_results) in enumerate(results.items()):
    h = pd.read_csv(method_results[0]).values[0:len(galaxy_classes), :]
    plt.plot(range(len(galaxy_classes)), np.sum(h * dum_values, axis=1) / np.sum(h, axis=1), label=method)

plt.legend()

#%%
for index, (method, method_results) in enumerate(results.items()):
    h = pd.read_csv(method_results[0]).values
    h_smooth = np.zeros((h.shape[0], int(h.shape[1]/method_results[1])))
    dum_smooth = np.zeros(int(len(dum_values) / method_results[1]))
    
    for s in range(method_results[1]):
        h_smooth += h[:, s::method_results[1]]
        dum_smooth += dum_values[s::method_results[1]]
    
    h_smooth /= method_results[1]
    dum_smooth /= method_results[1]

    #h_smooth = (
    #    h[:, 0::4] + h[:, 1::4] + h[:, 2::4] + h[:, 3::4]
    #) / 4
    
    for c in range(len(galaxy_classes)):
        plt.figure(c)
        plt.ylim((0.8, 1.2))
        plt.title("%s (%d samples)" % (
            galaxy_classes[c]["label"],
            h[c].sum() / 20000
        ))

        plt.plot(
            dum_smooth,
            h_smooth[c] / h_smooth[c].sum() / method_results[1] * 100,
            label="%s <%.4f>" % (
                method,
                np.sum(h[c] * dum_values) / np.sum(h[c])
            )
        )

    #plt.figure(len(sern_bins) + 1 + index)
    #plt.title(method)
    #plt.imshow(hist.values.T, origin="lower")

for c in range(len(galaxy_classes)):
    plt.figure(c)
    plt.gca().legend()
    #plt.savefig("plots/%s_results.png" % galaxy_classes[c]["label"])
