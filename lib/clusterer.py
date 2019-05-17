#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from lib import PDF, BayesianApproximation2d
from lib.plotting import *


class Clusterer:
    def __init__(self, galaxies, n_clusters):
        self.galaxies = galaxies

        X = np.column_stack((
            #galaxies["ba"],
            galaxies["sern"],
            galaxies["redshift"],
            galaxies["rmag"],
            galaxies["rabsmag"],
            galaxies["rad"]
        ))
        
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        
        km = KMeans(n_clusters=n_clusters)
        #km = GaussianMixture(n_components=n_clusters)
        km.fit(X_scaled)
        km.predict(X_scaled)

        self.inertia = km.inertia_

        # Sort labels by number of galaxies in cluster
        N = [len(galaxies[km.labels_ == i]) for i in range(n_clusters)]
        label_order = np.flip(np.argsort(N))
        self.labels = np.array([label_order[label] for label in km.labels_])

        # Generate ba for each cluster
        self.ba = [
            BayesianApproximation2d(PDF.from_samples(
                np.linspace(0, 1, 100),
                galaxies[self.labels == i]["ba"].values
            )) for i in range(n_clusters)
        ]
    
    def get_label(self, galaxy):
        return self.labels[self.galaxies.index[
            self.galaxies["id"] == galaxy["id"]
        ].tolist()[0]]
    
    def get_cluster(self, cluster):
        return self.galaxies[self.labels == cluster]


#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/galaxies.csv")
    test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")

#%%
if __name__ == "__main__":
    q_pdf = PDF.from_samples(np.linspace(0, 1, 100), galaxies["ba"])
    q_slots = test_galaxies["ba"].multiply(100).apply(np.ceil).astype(int) - 1

    np.sum(q_pdf.y[q_slots]) / len(test_galaxies) / 100

#%%
if __name__ == "__main__":
    inertia = []
    score = []

    for n_clusters in range(20, 21):
        clusterer = Clusterer(galaxies, n_clusters)

        p = []
        N = []

        for cluster_index in range(n_clusters):
            cluster = clusterer.get_cluster(cluster_index)
            test_cluster = test_galaxies[test_galaxies["id"].isin(cluster["id"])]

            if len(test_cluster) == 0:
                continue

            q_pdf = PDF.from_samples(
                np.linspace(0, 1, 100), cluster["ba"]
            )

            q_slots = test_cluster["ba"].multiply(100).apply(np.ceil).astype(int) - 1

            p.append(np.sum(q_pdf.y[q_slots]) / len(test_cluster) / 100)
            N.append(len(test_cluster))

        p = np.array(p)
        N = np.array(N)

        score.append((p * N).sum() / N.sum())
        inertia.append(clusterer.inertia)

        print(n_clusters, score[-1])

#%%
if __name__ == "__main__":
    #plt.plot(inertia, label="inertia")
    plt.plot(score, label="score")
    plt.legend()

#%%
if __name__ == "__main__":
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    fig, axes = plt.subplots(figsize=(16,6))
    bp = plt.boxplot(X_scaled)
    #plt.setp(bp['boxes'], color='black')
    #plt.setp(bp['whiskers'], color='black')
    #plt.setp(bp['fliers'], color='red', marker='o')
    plt.xlabel('Features')
    plt.ylabel('Value')
    axes.set_xticklabels(header, rotation=270)
    plt.grid()
