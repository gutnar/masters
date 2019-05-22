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
samples_per_galaxy = 50000
bootstrap_count = 100
bootstrap_sample_size = 1000
bootstrap_quantiles = (0.05, 0.5, 0.95)

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
    galaxies = galaxies[:100]

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
        
        print(np.array(hist))

        if len(hist) == 0:
            continue

        low, mean, high = np.quantile(np.array(hist), bootstrap_quantiles, axis=0)
        
        results["%s_low" % galaxy_class["label"]] = low / samples_per_galaxy / 2 * (len(dum_bins) - 1)
        results["%s_mean" % galaxy_class["label"]] = mean / samples_per_galaxy / 2 * (len(dum_bins) - 1)
        results["%s_high" % galaxy_class["label"]] = high / samples_per_galaxy / 2 * (len(dum_bins) - 1)

    print(time() - start)

    results.to_csv("data/final/%s_quantiles.csv" % method, index=False)
