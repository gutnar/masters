#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.cluster import KMeans

from lib import PDF, BayesianApproximation2d
from lib.plotting import *
from src.common import n_clusters

n_clusters = 8

#%%
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
e_spiral = pd.read_csv("data/raw/gama_spiral.txt", r"\s+")
e_elliptic = pd.read_csv("data/raw/gama_elliptic.txt", r"\s+")

e_spiral["e_class"] = 0
e_elliptic["e_class"] = 1

filament_galaxies = pd.concat((e_spiral, e_elliptic))
filament_galaxies = pd.merge(filament_galaxies, galaxies, on="id")

#%%
X = np.column_stack((
    galaxies["ba"],
    galaxies["sern"],
    galaxies["redshift"],
    galaxies["rmag"],
    galaxies["rabsmag"],
    galaxies["rad"]
))

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_filament = np.column_stack((
    filament_galaxies["ba"],
    filament_galaxies["sern"],
    filament_galaxies["redshift"],
    filament_galaxies["rmag"],
    filament_galaxies["rabsmag"],
    filament_galaxies["rad"]
))

#scaler = preprocessing.StandardScaler().fit(X_test)
X_filament_scaled = scaler.transform(X_filament)

#%%
km = KMeans(n_clusters=n_clusters)
km.fit(X_scaled)
labels = km.predict(X_scaled)
filament_labels = km.predict(X_filament_scaled)

# Sort labels by number of galaxies in cluster
# N = [len(galaxies[km.labels_ == i]) for i in range(n_clusters)]
# label_order = np.flip(np.argsort(N))
# labels = np.array([label_order[label] for label in km.labels_])

#%%
p = []
N = []

for i in range(n_clusters):
    cluster = galaxies[labels == i]
    filament_cluster = filament_galaxies[filament_labels == i]

    plt.hist(
        cluster["ba"], 100, (0, 1), True,
        histtype="step", label=str(len(cluster))
    )

    q_pdf = PDF.from_samples(np.linspace(0, 1, 100), cluster["ba"])
    q_slots = filament_cluster["ba"].multiply(100).apply(np.ceil).astype(int) - 1
    
    if len(filament_cluster):
        p.append(np.sum(q_pdf.y[q_slots]) / len(filament_cluster) / 100)
        N.append(len(filament_cluster))
    else:
        p.append(0)
        N.append(0)

plt.legend()

p = np.array(p)
N = np.array(N)

print(p, N)
print((p * N).sum() / N.sum())

#%%
galaxies["g_class"] = labels
filament_galaxies["g_class"] = filament_labels

galaxies.to_csv("data/intermediate/galaxies.csv", index=False)
filament_galaxies.to_csv("data/intermediate/filament_galaxies.csv", index=False)

#%% TEST
test = pd.read_csv("data/intermediate/galaxies.csv")

p = []
N = []

for i in range(n_clusters):
    cluster = test[test["g_class"] == i]
    q_pdf = PDF.from_samples(np.linspace(0, 1, 100), cluster["ba"])
    plt.plot(q_pdf.x, q_pdf.y, label=str(i) + ": " + str(len(cluster)))

plt.legend()
