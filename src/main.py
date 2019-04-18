#%%
import numpy as np
import pandas as pd
from time import time
from multiprocessing import Pool
from itertools import product
import os
import sys

sys.path.append(os.getcwd())

from lib import BayesianApproximation, PDF, get_dum
from tp_methods import *

#%%
processes = int(sys.argv[1])
samples_per_galaxy = int(sys.argv[2])
hist_bins = np.linspace(0, 1, 100)

#%%
if sys.argv[4] == "global":
    approximator = SampleApproximator(pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+"))
elif sys.argv[4] == "sample":
    approximator = SampleApproximator(pd.read_csv("data/intermediate/%s.csv" % sys.argv[3]))
elif sys.argv[4] == "classifier":
    approximator = ClassifierApproximator(pd.read_csv("data/intermediate/train_galaxies.csv"))
elif sys.argv[4] == "random":
    approximator = RandomApproximator()
elif sys.argv[4] == "classifier10":
    approximator = ClassifierApproximator(pd.read_csv("data/intermediate/train_galaxies.csv"),
        q_slot_multiplier=10,
        n_estimators=8,
        max_depth=28,
        max_features=5,
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True
    )

#%%
def process_galaxies(galaxies):
    dum_hist = np.zeros(len(hist_bins) - 1)
    
    for i, galaxy in galaxies.iterrows():
        t, p = approximator.sample_tp(galaxy, samples_per_galaxy)

        dum = np.concatenate(get_dum(
            np.repeat(galaxy["ra"], samples_per_galaxy),
            np.repeat(galaxy["dec"], samples_per_galaxy),
            p, t,
            np.repeat(galaxy["gama"], samples_per_galaxy),
            np.repeat(galaxy["ex"], samples_per_galaxy),
            np.repeat(galaxy["ey"], samples_per_galaxy),
            np.repeat(galaxy["ez"], samples_per_galaxy)
        ))

        dum_hist += np.histogram(dum, hist_bins)[0]

    return dum_hist

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/%s.csv" % sys.argv[3])
    #galaxies = galaxies[:4]

    pool = Pool(processes)
    chunks = np.array_split(galaxies, processes)

    start = time()
    dum_hist = np.sum(
        pool.map(process_galaxies, chunks), axis=0
    ) / len(galaxies) / samples_per_galaxy / 2
    print(time() - start)

    results = pd.DataFrame({
        "min": hist_bins[:-1],
        "max": hist_bins[1:],
        "N": dum_hist
    })

    results.to_csv("data/final/%s_%s.csv" % (sys.argv[3], sys.argv[4]), index=False)
