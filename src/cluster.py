#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from lib import PDF, BayesianApproximation2d
from lib.plotting import *
from src.common import n_clusters

#%%
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")

X = np.column_stack((
    galaxies["ba"],
    galaxies["sern"],
    galaxies["redshift"],
    galaxies["rmag"],
    galaxies["rabsmag"],
    galaxies["rad"]
))

#%%
km = KMeans(n_clusters=n_clusters)
km.fit(X)
km.predict(X)

# Sort labels by number of galaxies in cluster
N = [len(galaxies[km.labels_ == i]) for i in range(n_clusters)]
label_order = np.flip(np.argsort(N))
labels = np.array([label_order[label] for label in km.labels_])

for i in range(n_clusters):
    plt.figure(i)
    plt.hist(galaxies[labels == i]["ba"], 100)

#%%
for i in range(n_clusters):
    ba = BayesianApproximation2d(PDF.from_samples(
        np.linspace(0, 1, 100),
        galaxies[labels == i]["ba"].values
    ))
    ba.run()

    plt.figure(i*2)
    plot_ba_2d_results(ba)

    plt.figure(i*2 + 1)
    plot_xz_kde(ba)

#%%
galaxies["g_class"] = labels
galaxies.to_csv("data/intermediate/galaxies.csv", index=False)

#%%
e_spiral = pd.read_csv("data/raw/gama_spiral.txt", r"\s+")
e_elliptic = pd.read_csv("data/raw/gama_elliptic.txt", r"\s+")

e_spiral["e_class"] = 0
e_elliptic["e_class"] = 1

test_galaxies = pd.concat((e_spiral, e_elliptic))
test_galaxies = pd.merge(test_galaxies, galaxies, on="id")

test_galaxies.to_csv("data/intermediate/test_galaxies.csv", index=False)
