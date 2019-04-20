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

from lib import BayesianApproximation, PDF, get_dum
from tp_methods import *

#%%
processes = int(sys.argv[1])
samples_per_galaxy = 10000

max_sern = 20
sern_bins = np.linspace(0, max_sern, 51)
dum_bins = np.linspace(0, 1, 101)

#%%
if sys.argv[3] == "global":
    approximator = SampleApproximator(pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+"))
elif sys.argv[3] == "sample":
    approximator = SampleApproximator(pd.read_csv("data/intermediate/%s.csv" % sys.argv[2]))
elif sys.argv[3] == "classifier":
    approximator = ClassifierApproximator(pd.read_csv("data/intermediate/train_galaxies.csv"))
elif sys.argv[3] == "random":
    approximator = RandomApproximator()

#%%
def process_galaxies(galaxies):
    hist = np.zeros((len(sern_bins) - 1, len(dum_bins) - 1))
    #dum_hist = np.zeros(len(dum_bins) - 1)
    
    for i, galaxy in galaxies.iterrows():
        t, p = approximator.sample_tp(galaxy, samples_per_galaxy)

        dum = np.concatenate(get_dum(
            np.repeat(galaxy["ra"], samples_per_galaxy),
            np.repeat(galaxy["dec"], samples_per_galaxy),
            p,
            t,
            np.repeat(galaxy["gama"], samples_per_galaxy),
            np.repeat(galaxy["ex"], samples_per_galaxy),
            np.repeat(galaxy["ey"], samples_per_galaxy),
            np.repeat(galaxy["ez"], samples_per_galaxy)
        ))

        hist[int(galaxy["sern"]/(max_sern+0.000000000001)*(len(sern_bins) - 1))] += np.histogram(dum, dum_bins)[0]

        #dum_hist += np.histogram(dum, dum_bins)[0]

    return hist

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/%s.csv" % sys.argv[2])
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

    results.to_csv("data/final/%s_%s.csv" % (sys.argv[2], sys.argv[3]), index=False)
    #results.to_csv("data/final/test.csv", index=False)
