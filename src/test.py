#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")

galaxies.describe()

#%%
X = np.column_stack((
    galaxies["ba"],
    galaxies["sern"],
    galaxies["redshift"],
    galaxies["rmag"],
    galaxies["rabsmag"],
    galaxies["rad"]
))

#%%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets

#%% KMeans
n_clusters = (2, 3, 4, 5, 6, 7, 8, 9, 10)
inertia = []

for n in n_clusters:
    km = KMeans(n_clusters=n)
    km.fit(X)
    #km.predict(X)
    #labels = km.labels_
    inertia.append(km.inertia_)

plt.plot(n_clusters, inertia)

#%%
n_clusters = 4

km = KMeans(n_clusters=n_clusters)
km.fit(X)
km.predict(X)

labels = km.labels_

#%% Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels, edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)

#%%
for i in range(n_clusters):
    plt.figure(i)
    plt.hist(galaxies[labels == i]["ba"], 100)

#%%
from lib import PDF, BayesianApproximation2d
from lib.plotting import *

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
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
filament_galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")

filament_galaxies.describe()
