import numpy as np
from lib import BayesianApproximation, PDF


class SampleApproximator:
    def __init__(self, galaxies):
        q_pdf = PDF.from_samples(
            np.linspace(0, 1, 100),
            galaxies["ba"].values
        )

        self.ba = BayesianApproximation(q_pdf)
        self.ba.run()#([(1000, "scott")])
    
    def get_t_pdf(self, galaxy):
        return self.ba.get_t_pdf(galaxy["ba"])
    
    def get_p_pdf(self, galaxy):
        return self.ba.get_p_pdf(galaxy["ba"])


class ClassifierApproximator:
    def __init__(self):
        pass
