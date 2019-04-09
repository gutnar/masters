import numpy as np
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
        t_pdf = self.ba.get_t_pdf(galaxy["ba"])
        p_pdf = self.ba.get_p_pdf(galaxy["ba"])

        return np.column_stack((
            t_pdf.sample(N),
            p_pdf.sample(N)
        ))


class ClassifierApproximator:
    def __init__(self, galaxies):
        self.classifier = Classifier(100)
        self.classifier.fit(galaxies)

    def sample_tp(self, galaxy, N):
        q_pdf = self.classifier.predict_pdf(galaxy)

        ba = BayesianApproximation(q_pdf)
        ba.run()

        t_pdf = ba.get_t_pdf(galaxy["ba"])
        p_pdf = ba.get_p_pdf(galaxy["ba"])
        
        return np.column_stack((
            t_pdf.sample(N),
            p_pdf.sample(N)
        ))
