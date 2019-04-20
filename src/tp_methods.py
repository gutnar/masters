import numpy as np
import math
from lib import BayesianApproximation, PDF, Classifier


class SampleApproximator:
    def __init__(self, galaxies):
        q_pdf = PDF.from_samples(
            np.linspace(0, 1, 100),
            galaxies["ba"].values
        )

        self.ba = BayesianApproximation(q_pdf)
        self.ba.run()
    
    def sample_tp(self, galaxy, N):
        return self.ba.sample_tp(galaxy["ba"], N)


class ClassifierApproximator:
    def __init__(self, galaxies, **kwargs):
        self.classifier = Classifier(**kwargs)
        self.classifier.fit(galaxies)

    def sample_tp(self, galaxy, N):
        q_pdf = self.classifier.predict_pdf(galaxy)

        ba = BayesianApproximation(q_pdf)
        ba.run()

        return ba.sample_tp(galaxy["ba"], N)


class RandomApproximator:
    def sample_tp(self, galaxy, N):
        return (
            np.concatenate((
                -np.arccos(np.random.uniform(-1, 1, math.floor(N/2))),
                np.arccos(np.random.uniform(-1, 1, math.ceil(N/2)))
            )),
            np.random.uniform(-np.pi/2, np.pi/2, N)
        )
