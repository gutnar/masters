#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.cluster import KMeans

from lib import PDF, BayesianApproximation2d
from lib.plotting import *
from src.common import n_clusters

#%%
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
e_spiral = pd.read_csv("data/raw/gama_spiral.txt", r"\s+")
e_elliptic = pd.read_csv("data/raw/gama_elliptic.txt", r"\s+")

e_spiral["e_class"] = 0
e_elliptic["e_class"] = 1

test_galaxies = pd.concat((e_spiral, e_elliptic))
test_galaxies = pd.merge(test_galaxies, galaxies, on="id")

train_galaxies = galaxies[~galaxies["id"].isin(test_galaxies["id"])]

#%%
X = np.column_stack((
    train_galaxies["ba"],
    train_galaxies["sern"],
    train_galaxies["redshift"],
    train_galaxies["rmag"],
    train_galaxies["rabsmag"],
    train_galaxies["rad"]
))

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_test = np.column_stack((
    test_galaxies["ba"],
    test_galaxies["sern"],
    test_galaxies["redshift"],
    test_galaxies["rmag"],
    test_galaxies["rabsmag"],
    test_galaxies["rad"]
))

scaler = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)

#%%
km = KMeans(n_clusters=n_clusters)
km.fit(X_scaled)
labels = km.predict(X_scaled)
test_labels = km.predict(X_test_scaled)

# Sort labels by number of galaxies in cluster
# N = [len(galaxies[km.labels_ == i]) for i in range(n_clusters)]
# label_order = np.flip(np.argsort(N))
# labels = np.array([label_order[label] for label in km.labels_])

#%%
p = []
N = []

for i in range(n_clusters):
    cluster = train_galaxies[labels == i]
    test_cluster = test_galaxies[test_labels == i]

    plt.hist(
        cluster["ba"], 100, (0, 1), True,
        histtype="step", label=str(len(cluster))
    )

    q_pdf = PDF.from_samples(np.linspace(0, 1, 100), cluster["ba"])
    q_slots = test_cluster["ba"].multiply(100).apply(np.ceil).astype(int) - 1
    
    if len(test_cluster):
        p.append(np.sum(q_pdf.y[q_slots]) / len(test_cluster) / 100)
        N.append(len(test_cluster))
    else:
        p.append(0)
        N.append(0)

plt.legend()

p = np.array(p)
N = np.array(N)

print(p, N)
print((p * N).sum() / N.sum())

#%%
cluster = galaxies[labels == 1]

ba = BayesianApproximation2d(PDF.from_samples(
    np.linspace(0, 1, 100),
    cluster["ba"].values
))

ba.run()

plt.figure(1)
plot_ba_2d_results(ba)

plt.figure(2)
plot_xz_kde(ba)

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
test_galaxies.to_csv("data/intermediate/test_galaxies.csv", index=False)
