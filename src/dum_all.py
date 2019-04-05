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

#%%
if __name__ != "__main__":
    galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")

    q_pdf = PDF.from_samples(
        np.linspace(0, 1, 100),
        galaxies["ba"].values
    )

    ba = BayesianApproximation(q_pdf)
    ba.run([(1000, "scott")])

#%%
def process_galaxies(galaxies):
    t_samples = []
    p_samples = []

    for i, galaxy in galaxies.iterrows():
        t_pdf = ba.get_t_pdf(galaxy["ba"])
        p_pdf = ba.get_p_pdf(galaxy["ba"])

        t_samples.append(t_pdf.sample(10))
        p_samples.append(p_pdf.sample(10))

    return t_samples, p_samples

#%%
if __name__ == "__main__":
    gama = pd.read_csv("data/raw/gama_data_for_gutnar.txt", r"\s+")
    inclinations = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
    galaxies = gama.merge(inclinations, on="id", how="inner")
    galaxies = galaxies[:10]

    process_count = 2
    pool = Pool(process_count)
    chunks = np.array_split(galaxies, process_count)

    start = time()
    results = np.vstack(pool.map(process_galaxies, chunks))
    print(time() - start)

    t = results[:,0,:].flatten()
    p = results[:,1,:].flatten()

    plt.figure(1)
    plt.hist(np.cos(t), 100, (0, 1), True)
    plt.savefig("plots/test_cos_t.png")

    plt.figure(2)
    plt.hist(p, 100, (0, np.pi), True)
    plt.savefig("plots/test_p.png")

    pd.DataFrame(results[:,0,:]).to_csv("data/intermediate/test.txt")