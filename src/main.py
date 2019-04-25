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
samples_per_galaxy = 10000

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
elif method == "kmeans":
    approximator = BaApproximator()
elif method == "kmeans1d":
    approximator = BaApproximator()

#%%
def process_galaxies(galaxies):
    hist = np.zeros((len(galaxy_classes), len(dum_bins) - 1))
    
    for i, galaxy in galaxies.iterrows():
        pos, inc = approximator.sample_pos_inc(galaxy, samples_per_galaxy)

        dum = np.concatenate(get_dum(
            np.repeat(galaxy["ra"], samples_per_galaxy),
            np.repeat(galaxy["dec"], samples_per_galaxy),
            pos,
            inc,
            np.repeat(galaxy["gama"], samples_per_galaxy),
            np.repeat(galaxy["ex"], samples_per_galaxy),
            np.repeat(galaxy["ey"], samples_per_galaxy),
            np.repeat(galaxy["ez"], samples_per_galaxy)
        ))

        dum_hist = np.histogram(dum, dum_bins)[0]

        for c in range(len(galaxy_classes)):
            if galaxy[galaxy_classes[c]["parameter"]] == galaxy_classes[c]["value"]:
                hist[c] += dum_hist

    return hist

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")
    #galaxies = galaxies[:4]

    pool = Pool(processes)
    chunks = np.array_split(galaxies, processes)

    start = time()
    hist = np.sum(
        pool.map(process_galaxies, chunks), axis=0
    )
    print(time() - start)

    #results = pd.DataFrame({
    #    "min": hist_bins[:-1],
    #    "max": hist_bins[1:],
    #    "N": dum_hist
    #})
    results = pd.DataFrame(hist)

    results.to_csv("data/final/%s.csv" % method, index=False)
    #results.to_csv("data/final/test.csv", index=False)
