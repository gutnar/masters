#%%
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
from scipy.stats import truncnorm

from lib import *
from src.common import n_clusters


def get_truncnorm_sample(mu, sigma, a, b, N):
    return truncnorm.rvs((a - mu)/sigma, (b - mu)/sigma, loc=mu, scale=sigma, size=N)


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


class PosApproximator:
    def sample_pos_inc(self, galaxy, N):
        return (
            np.repeat(galaxy["pos"]/180*np.pi, N),
            np.random.uniform(0, galaxy["ba"], N)
        )


class SpiralPosApproximator:
    def sample_pos_inc(self, galaxy, N):
        f = get_truncnorm_sample(0.222, 0.057, 0, galaxy["ba"], N)

        return (
            np.repeat(galaxy["pos"]/180*np.pi, N),
            np.sqrt((galaxy["ba"]**2 - f**2) / (1 - f**2))
        )


class EllipticPosApproximator:
    def sample_pos_inc(self, galaxy, N):
        f = get_truncnorm_sample(0.7, 0.1, 0, galaxy["ba"], N)

        return (
            np.repeat(galaxy["pos"]/180*np.pi, N),
            np.sqrt((galaxy["ba"]**2 - f**2) / (1 - f**2))
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
            galaxy["pos"]/180*np.pi
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
    def __init__(self, methods=[(150000, "scott")]*50, **kwargs):
        train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")
        test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")
        
        self.classifier = Classifier(**kwargs)
        self.classifier.fit(train_galaxies)

        self.methods = methods

    def sample_pos_inc(self, galaxy, N):
        q_pdf = self.classifier.predict_pdf(galaxy)

        ba = BayesianApproximation2d(q_pdf)
        ba.run(self.methods)

        return ba.sample_pos_inc(
            galaxy["ba"],
            galaxy["pos"]/180*np.pi
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
    def __init__(self, methods=[(150000, 0.05)]*50):
        galaxies = pd.read_csv("data/intermediate/galaxies.csv")

        self.ba = [
            BayesianApproximation2d(PDF.from_samples(
                np.linspace(0, 1, 100),
                galaxies[galaxies["g_class"] == c]["ba"].values
            )) for c in range(n_clusters)
        ]

        self.ready = [False] * n_clusters
        self.methods = methods
    
    def sample_pos_inc(self, galaxy, N):
        g_class = int(galaxy["g_class"])

        if not self.ready[g_class]:
            self.ba[g_class].run(self.methods)
            self.ready[g_class] = True
        
        return self.ba[g_class].sample_pos_inc(
            galaxy["ba"],
            galaxy["pos"]/180*np.pi
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


class RydenApproximator:
    def __init__(self):
        self.ba = BayesianApproximation2d(None)

        xi = np.random.normal(0.222, 0.057, 150000)
        zeta = 1 - np.exp(np.random.normal(-1.85, 0.89, 150000))
        valid = xi < zeta
        xi = xi[valid]
        zeta = zeta[valid]
        
        self.ba.xz_kde = stats.kde.gaussian_kde(np.column_stack((xi, zeta)).T)

    def sample_pos_inc(self, galaxy, N):
        return self.ba.sample_pos_inc(
            galaxy["ba"],
            galaxy["pos"]/180*np.pi
        )


class BoschVenApproximator:
    def __init__(self):
        self.ba = BayesianApproximation2d(None)

        xi = np.random.normal(0.7, 0.1, 150000)
        zeta = 1 - np.exp(np.random.normal(-1.85, 0.89, 150000))
        valid = (xi < zeta) & (zeta > 0.7) & (zeta < 0.9)
        xi = xi[valid]
        zeta = zeta[valid]
        
        self.ba.xz_kde = stats.kde.gaussian_kde(np.column_stack((xi, zeta)).T)

    def sample_pos_inc(self, galaxy, N):
        return self.ba.sample_pos_inc(
            galaxy["ba"],
            galaxy["pos"]/180*np.pi
        )
