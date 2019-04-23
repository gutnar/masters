import numpy as np
import math
from lib import *


class SampleApproximator:
    def __init__(self, galaxies):
        q_pdf = PDF.from_samples(
            np.linspace(0, 1, 100),
            galaxies["ba"].values
        )

        self.ba = BayesianApproximation(q_pdf)
        self.ba.run()
    
    def sample_pos_inc(self, galaxy, N):
        return self.ba.sample_pos_inc(galaxy["ba"], galaxy["pos"], N)


class SampleApproximator1d:
    def __init__(self, galaxies):
        q_pdf = PDF.from_samples(
            np.linspace(0, 1, 100),
            galaxies["ba"].values
        )

        self.ba = BayesianApproximation1d(q_pdf)
        self.ba.run()
    
    def sample_pos_inc(self, galaxy, N):
        return (
            np.repeat(galaxy["pos"], N),
            self.ba.get_i_pdf(galaxy["ba"]).sample(N)
        )


class ClassifierApproximator:
    def __init__(self, galaxies, **kwargs):
        self.classifier = Classifier(**kwargs)
        self.classifier.fit(galaxies)

    def sample_pos_inc(self, galaxy, N):
        q_pdf = self.classifier.predict_pdf(galaxy)

        ba = BayesianApproximation(q_pdf)
        ba.run()

        return ba.sample_pos_inc(galaxy["ba"], galaxy["pos"], N)


class ClassifierApproximator1d:
    def __init__(self, galaxies, **kwargs):
        self.classifier = Classifier(**kwargs)
        self.classifier.fit(galaxies)

    def sample_pos_inc(self, galaxy, N):
        q_pdf = self.classifier.predict_pdf(galaxy)

        ba = BayesianApproximation1d(q_pdf)
        ba.run()

        return (
            np.repeat(galaxy["pos"], N),
            ba.get_i_pdf(galaxy["ba"]).sample(N)
        )


class RandomApproximator:
    def sample_pos_inc(self, galaxy, N):
        return (
            np.random.uniform(-np.pi/2, np.pi/2, N),
            np.random.uniform(0, 1, N)
        )
