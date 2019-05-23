#%%
import numpy as np
import pandas as pd
from time import time
from multiprocessing import Pool
from itertools import product
from sklearn.preprocessing import normalize
import os
import sys

sys.path.append(os.getcwd())

from src.approximators import *
from src.common import n_clusters, galaxy_classes

#%%
processes = int(sys.argv[1])
samples_per_galaxy = 5000
bootstrap_count = 1000
bootstrap_quantiles = (0.025, 0.5, 0.975)
dum_bins = np.linspace(0, 1, int(sys.argv[3]) + 1)

#%%
method = sys.argv[2]

if method == "global":
    approximator = GlobalApproximator()
elif method == "global1d":
    approximator = Global1dApproximator()
elif method == "sern":
    approximator = SernApproximator()
elif method == "rf":
    approximator = RandomForestApproximator()
elif method == "rf1d":
    approximator = RandomForest1dApproximator()
elif method == "random":
    approximator = RandomApproximator()
elif method == "ba":
    approximator = BaApproximator()
elif method == "pos":
    approximator = PosApproximator()
elif method == "spiral_pos":
    approximator = SpiralPosApproximator()
elif method == "elliptic_pos":
    approximator = EllipticPosApproximator()
elif method == "kmeans":
    approximator = KMeansApproximator()
elif method == "kmeans1d":
    approximator = KMeans1dApproximator()
elif method == "ryden":
    approximator = RydenApproximator()
elif method == "bosch_ven":
    approximator = BoschVenApproximator()

#%%
def process_galaxies(galaxies):
    j = [0] * len(galaxy_classes)
    weights = [
        np.zeros(sum(galaxies[c["parameter"]] == c["value"])) for c in galaxy_classes
    ]
    hists = [
        np.zeros((sum(galaxies[c["parameter"]] == c["value"]), len(dum_bins) - 1)) for c in galaxy_classes
    ]
    
    for i, galaxy in galaxies.iterrows():
        pos, inc = approximator.sample_pos_inc(galaxy, samples_per_galaxy)
        N = len(pos)

        galaxy_hist = np.histogram(np.concatenate(get_dum(
            np.repeat(galaxy["ra"], N),
            np.repeat(galaxy["dec"], N),
            pos,
            inc,
            np.repeat(galaxy["gama"], N),
            np.repeat(galaxy["ex"], N),
            np.repeat(galaxy["ey"], N),
            np.repeat(galaxy["ez"], N)
        )), dum_bins, density=True)[0]

        for c in range(len(galaxy_classes)):
            if galaxy[galaxy_classes[c]["parameter"]] == galaxy_classes[c]["value"]:
                weights[c][j[c]] = N
                hists[c][j[c],:] = galaxy_hist
                j[c] += 1

    return {
        "weights": weights,
        "hists": hists
    }

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")
    #galaxies = galaxies[:4]

    pool = Pool(processes)
    chunks = np.array_split(galaxies, processes)

    start = time()
    hist_raw = pool.map(process_galaxies, chunks)
    print(time() - start)

    results = pd.DataFrame({
        "dum_min": dum_bins[:-1],
        "dum_mean": (dum_bins[:-1] + dum_bins[1:]) / 2,
        "dum_max": dum_bins[1:]
    })

    start = time()

    for c, galaxy_class in enumerate(galaxy_classes):
        weight = []
        hist = []

        for process in range(processes):
            weight += list(hist_raw[process]["weights"][c])
            hist += list(hist_raw[process]["hists"][c])
        
        weight = np.array(weight) / np.sum(weight)
        hist = np.array(hist)
        print(weight.shape, hist.shape, np.sum(weight))

        if len(hist) == 0:
            continue
        
        hist_mean = []
        hist_std = []

        for i in range(len(dum_bins) - 1):
            i_means = [
                np.mean(np.random.choice(hist[:,i], hist.shape[0]))
                for j in range(bootstrap_count)
            ]

            hist_mean.append(np.mean(i_means))
            hist_std.append(np.std(i_means))
        
        results["%s_mean" % galaxy_class["label"]] = np.array(hist_mean)
        results["%s_std" % galaxy_class["label"]] = np.array(hist_std)

    print(time() - start)

    results.to_csv("data/final/%s_quantiles.csv" % method, index=False)
