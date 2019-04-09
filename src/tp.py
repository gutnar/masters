#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
from itertools import product
import os
import sys

sys.path.append(os.getcwd())

from lib import BayesianApproximation, PDF
from tp_methods import SampleApproximator, ClassifierApproximator

#%%
processes = int(sys.argv[1])
samples_per_galaxy = int(sys.argv[2])

#%%
if __name__ != "__main__":
    if sys.argv[4] == "global":
        approximator = SampleApproximator(pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+"))
    elif sys.argv[4] == "sample":
        approximator = SampleApproximator(pd.read_csv("data/intermediate/%s.csv" % sys.argv[3]))
    elif sys.argv[4] == "classifier":
        approximator = ClassifierApproximator()

#%%
def process_galaxies(galaxies):
    t_samples = []
    p_samples = []

    for i, galaxy in galaxies.iterrows():
        t_pdf = approximator.get_t_pdf(galaxy)
        p_pdf = approximator.get_p_pdf(galaxy)

        t_samples.append(t_pdf.sample(samples_per_galaxy))
        p_samples.append(p_pdf.sample(samples_per_galaxy))

    return t_samples, p_samples

#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/%s.csv" % sys.argv[3])

    pool = Pool(processes)
    chunks = np.array_split(galaxies, processes)

    start = time()
    results = np.array(pool.map(process_galaxies, chunks))
    print(time() - start)

    t = results[:,0,:,:].flatten()
    p = results[:,1,:,:].flatten()

    plt.figure(1)
    plt.hist(np.cos(t), 100, (0, 1), True)
    plt.savefig("plots/%s_%s_cos_t.png" % (sys.argv[3], sys.argv[4]))

    plt.figure(2)
    plt.hist(p, 100, (0, np.pi), True)
    plt.savefig("plots/%s_%s_p.png" % (sys.argv[3], sys.argv[4]))

    galaxies = galaxies.loc[galaxies.index.repeat(samples_per_galaxy)]
    galaxies["t"] = t
    galaxies["p"] = p
    galaxies.to_csv("data/intermediate/%s_%s.csv" % (sys.argv[3], sys.argv[4]), index=False)
