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
    dum_mean = []
    
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

        dum_mean.append(np.mean(dum))
    
    return np.array(dum_mean)

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/%s.csv" % sys.argv[2])

    pool = Pool(processes)
    chunks = np.array_split(galaxies, processes)

    start = time()
    galaxies["dum_mean"] = np.concatenate(
        pool.map(process_galaxies, chunks), axis=0
    )
    print(time() - start)

    galaxies.to_csv("data/final/%s_%s_dum_mean.csv" % (sys.argv[2], sys.argv[3]), index=False)
