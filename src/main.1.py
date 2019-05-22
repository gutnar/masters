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
from src.common import n_clusters, galaxy_classes, dum_bins

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
elif method == "kmeans_scott":
    approximator = KMeansApproximator([(150000, "scott")]*50)
elif method == "rf_scott":
    approximator = RandomForestApproximator([150000, "scott"]*50)
elif method == "ryden":
    approximator = RydenApproximator()
elif method == "bosch_ven":
    approximator = BoschVenApproximator()

#%%
def process_galaxies(galaxies):
    j = [0, 0]
    hists = [
        np.zeros((sum(galaxies[c["parameter"]] == c["value"]), len(dum_bins) - 1)) for c in galaxy_classes
    ]
    
    for i, galaxy in galaxies.iterrows():
        pos, inc = approximator.sample_pos_inc(galaxy, samples_per_galaxy)

        galaxy_hist = np.histogram(np.concatenate(get_dum(
            np.repeat(galaxy["ra"], samples_per_galaxy),
            np.repeat(galaxy["dec"], samples_per_galaxy),
            pos,
            inc,
            np.repeat(galaxy["gama"], samples_per_galaxy),
            np.repeat(galaxy["ex"], samples_per_galaxy),
            np.repeat(galaxy["ey"], samples_per_galaxy),
            np.repeat(galaxy["ez"], samples_per_galaxy)
        )), dum_bins)[0]

        for c in range(len(galaxy_classes)):
            if galaxy[galaxy_classes[c]["parameter"]] == galaxy_classes[c]["value"]:
                hists[c][j[c],:] = galaxy_hist
                j[c] += 1

    return hists

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
        hist = []

        for process in range(processes):
            hist += list(hist_raw[process][c])
        
        hist = np.array(hist)
        print(hist.shape)

        if len(hist) == 0:
            continue
        
        hist_mean = []
        hist_std = []

        for i in range(len(dum_bins) - 1):
            i_means = [np.mean(np.random.choice(hist[:,i], hist.shape[0])) for j in range(bootstrap_count)]

            hist_mean.append(np.mean(i_means))
            hist_std.append(np.std(i_means))
        
        results["%s_mean" % galaxy_class["label"]] = np.array(hist_mean) / samples_per_galaxy / 2 * (len(dum_bins) - 1)
        results["%s_std" % galaxy_class["label"]] = np.array(hist_std) / samples_per_galaxy / 2 * (len(dum_bins) - 1)

    print(time() - start)

    results.to_csv("data/final/%s_quantiles.1.csv" % method, index=False)
