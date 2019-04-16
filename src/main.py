#%%
import numpy as np
import pandas as pd
from time import time
from multiprocessing import Pool
from itertools import product
import os
import sys

sys.path.append(os.getcwd())

from lib import BayesianApproximation, PDF
from tp_methods import *

#%%
processes = int(sys.argv[1])
samples_per_galaxy = int(sys.argv[2])

#%%
if sys.argv[4] == "global":
    approximator = SampleApproximator(pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+"))
elif sys.argv[4] == "sample":
    approximator = SampleApproximator(pd.read_csv("data/intermediate/%s.csv" % sys.argv[3]))
elif sys.argv[4] == "classifier":
    approximator = ClassifierApproximator(pd.read_csv("data/intermediate/train_galaxies.csv"))
elif sys.argv[4] == "random":
    approximator = RandomApproximator()

#%%
def process_galaxies(galaxies):
    hist = np.zeros(100)
    samples = np.empty((0, 2))
    
    for i, galaxy in galaxies.iterrows():
        t, p = approximator.sample_tp(galaxy, samples_per_galaxy)
        
        galaxy = gals.loc[np.repeat(1, N)]

        dum = np.concatenate(get_dum(
            galaxy["ra"], galaxy["dec"],
            p_pdf.sample(N), t_pdf.sample(N),
            galaxy["gama"],
            galaxy["ex"], galaxy["ey"], galaxy["ez"]
        ))

        dum_hist = np.zeros(100)
        dum_hist += np.histogram(dum, 100, (0, 1))[0]

        plt.plot(dum_hist)

    return samples

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/%s.csv" % sys.argv[3])
    #galaxies = galaxies[:10]

    pool = Pool(processes)
    chunks = np.array_split(galaxies, processes)

    start = time()
    results = np.vstack(pool.map(process_galaxies, chunks))
    print(time() - start)

    #print(results)
    print(results.shape)

    t = results[:,0]
    p = results[:,1]

    galaxies = galaxies.loc[galaxies.index.repeat(samples_per_galaxy)]
    galaxies["t"] = t
    galaxies["p"] = p
    galaxies.to_csv("data/intermediate/%s_%s.csv" % (sys.argv[3], sys.argv[4]), index=False)
