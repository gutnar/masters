import pandas as pd
import numpy as np
import math

from lib import *
from src.common import n_clusters


class RandomApproximator:
    def sample_pos_inc(self, galaxy, N):
        return (
            np.random.uniform(-np.pi/2, np.pi/2, N),
            np.random.uniform(0, 1, N)
        )


class BaApproximator:
    def sample_pos_inc(self, galaxy, N):
        return (
            np.repeat(galaxy["pos"]/180*np.pi, N),
            np.repeat(galaxy["ba"], N)
        )


class GlobalApproximator:
    def __init__(self):
        galaxies = pd.read_csv("data/intermediate/galaxies.csv")

        q_pdf = PDF.from_samples(
            np.linspace(0, 1, 100),
            galaxies["ba"].values
        )

        self.ba = BayesianApproximation2d(q_pdf)
        self.ready = False
    
    def sample_pos_inc(self, galaxy, N):
        if not self.ready:
            self.ba.run()
            self.ready = True
        
        return self.ba.sample_pos_inc(
            galaxy["ba"],
            galaxy["pos"]/180*np.pi,
            N
        )


class Global1dApproximator:
    def __init__(self):
        galaxies = pd.read_csv("data/intermediate/galaxies.csv")

        q_pdf = PDF.from_samples(
            np.linspace(0, 1, 100),
            galaxies["ba"].values
        )

        self.ba = BayesianApproximation1d(q_pdf)
        self.ready = False
    
    def sample_pos_inc(self, galaxy, N):
        if not self.ready:
            self.ba.run()
            self.ready = True
        
        return (
            np.repeat(galaxy["pos"]/180*np.pi, N),
            self.ba.get_i_pdf(galaxy["ba"]).sample(N)
        )


class RandomForestApproximator:
    def __init__(self, **kwargs):
        galaxies = pd.read_csv("data/intermediate/galaxies.csv")

        self.classifier = Classifier(**kwargs)
        self.classifier.fit(galaxies)

    def sample_pos_inc(self, galaxy, N):
        q_pdf = self.classifier.predict_pdf(galaxy)

        ba = BayesianApproximation2d(q_pdf)
        ba.run()

        return ba.sample_pos_inc(
            galaxy["ba"],
            galaxy["pos"]/180*np.pi,
            N
        )


class RandomForest1dApproximator:
    def __init__(self, **kwargs):
        galaxies = pd.read_csv("data/intermediate/galaxies.csv")

        self.classifier = Classifier(**kwargs)
        self.classifier.fit(galaxies)

    def sample_pos_inc(self, galaxy, N):
        q_pdf = self.classifier.predict_pdf(galaxy)

        ba = BayesianApproximation1d(q_pdf)
        ba.run()

        return (
            np.repeat(galaxy["pos"]/180*np.pi, N),
            ba.get_i_pdf(galaxy["ba"]).sample(N)
        )


class KMeansApproximator:
    def __init__(self):
        galaxies = pd.read_csv("data/intermediate/galaxies.csv")

        self.ba = [
            BayesianApproximation2d(PDF.from_samples(
                np.linspace(0, 1, 100),
                galaxies[galaxies["g_class"] == c]["ba"].values
            )) for c in range(n_clusters)
        ]

        self.ready = [False] * n_clusters
    
    def sample_pos_inc(self, galaxy, N):
        g_class = int(galaxy["g_class"])

        if not self.ready[g_class]:
            self.ba[g_class].run()
            self.ready[g_class] = True
        
        return self.ba[g_class].sample_pos_inc(
            galaxy["ba"],
            galaxy["pos"]/180*np.pi,
            N
        )


class KMeans1dApproximator:
    def __init__(self):
        galaxies = pd.read_csv("data/intermediate/galaxies.csv")

        self.ba = [
            BayesianApproximation1d(PDF.from_samples(
                np.linspace(0, 1, 100),
                galaxies[galaxies["g_class"] == c]["ba"].values
            )) for c in range(n_clusters)
        ]

        self.ready = [False] * n_clusters
    
    def sample_pos_inc(self, galaxy, N):
        g_class = int(galaxy["g_class"])

        if not self.ready[g_class]:
            self.ba[g_class].run()
            self.ready[g_class] = True

        return (
            np.repeat(galaxy["pos"]/180*np.pi, N),
            self.ba[g_class].get_i_pdf(galaxy["ba"]).sample(N)
        )
