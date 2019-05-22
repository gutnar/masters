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
samples_per_galaxy = 500
bootstrap_count = 100
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
    #hist_low = np.zeros((len(galaxy_classes), len(dum_bins) - 1))
    #hist_mean = np.zeros((len(galaxy_classes), len(dum_bins) - 1))
    #hist_high = np.zeros((len(galaxy_classes), len(dum_bins) - 1))

    dum = [[] for c in galaxy_classes]
    
    for i, galaxy in galaxies.iterrows():
        pos, inc = approximator.sample_pos_inc(galaxy, samples_per_galaxy)

        galaxy_dum = list(np.concatenate(get_dum(
            np.repeat(galaxy["ra"], samples_per_galaxy),
            np.repeat(galaxy["dec"], samples_per_galaxy),
            pos,
            inc,
            np.repeat(galaxy["gama"], samples_per_galaxy),
            np.repeat(galaxy["ex"], samples_per_galaxy),
            np.repeat(galaxy["ey"], samples_per_galaxy),
            np.repeat(galaxy["ez"], samples_per_galaxy)
        )))

        #dum_hists = np.zeros((bootstrap_count, len(dum_bins) - 1))

        #for i in range(bootstrap_count):
        #    dum_sample = np.random.choice(dum, samples_per_galaxy)
        #    dum_hists[i,:] = np.histogram(dum_sample, dum_bins)[0]
        
        #dum_low, dum_mean, dum_high = np.quantile(dum_hists, bootstrap_quantiles, axis=0)

        for c in range(len(galaxy_classes)):
            if galaxy[galaxy_classes[c]["parameter"]] == galaxy_classes[c]["value"]:
                dum[c] += galaxy_dum
                #hist_low[c] += dum_low
                #hist_mean[c] += dum_mean
                #hist_high[c] += dum_high

    return dum#hist_low, hist_mean, hist_high

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")
    #galaxies = galaxies[:4]

    pool = Pool(processes)
    chunks = np.array_split(galaxies, processes)

    start = time()
    dum = [sum(d, []) for d in np.array(pool.map(process_galaxies, chunks)).T]
    print(time() - start)

    #results = pd.DataFrame({
    #    "min": hist_bins[:-1],
    #    "max": hist_bins[1:],
    #    "N": dum_hist
    #})
    #results = pd.DataFrame(mean)

    results = pd.DataFrame({
        "dum_min": dum_bins[:-1],
        "dum_mean": (dum_bins[:-1] + dum_bins[1:]) / 2,
        "dum_max": dum_bins[1:]
    })

    start = time()

    for c in range(len(galaxy_classes)):
        dum_hists = np.zeros((bootstrap_count, len(dum_bins) - 1))
        dum_mean = []

        for i in range(bootstrap_count):
            dum_sample = np.random.choice(dum[c], len(dum[c]))
            dum_mean.append(np.mean(dum_sample))
            dum_hists[i,:] = np.histogram(dum_sample, dum_bins)[0]
        
        dum_hist_low, dum_hist_mean, dum_hist_high = np.quantile(dum_hists, bootstrap_quantiles, axis=0)

        results["%s_low" % galaxy_classes[c]["label"]] = dum_hist_low / len(dum[c]) * (len(dum_bins) - 1)
        results["%s_mean" % galaxy_classes[c]["label"]] = dum_hist_mean / len(dum[c]) * (len(dum_bins) - 1)
        results["%s_high" % galaxy_classes[c]["label"]] = dum_hist_high / len(dum[c]) * (len(dum_bins) - 1)
        results["%s_mu" % galaxy_classes[c]["label"]] = np.mean(dum_mean)
        results["%s_sigma" % galaxy_classes[c]["label"]] = np.std(dum_mean)
    
    print(time() - start)

    results.to_csv("data/final/%s_quantiles.csv" % method, index=False)
    #results.to_csv("data/final/test.csv", index=False)
