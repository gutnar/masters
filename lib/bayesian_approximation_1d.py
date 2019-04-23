#%%
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import random
import math

from lib.pdf import PDF


class BayesianApproximation1d:
    @classmethod
    def __init__(self, q_pdf, size=10000):
        self.q_pdf = q_pdf

        # Initialize kde
        q = q_pdf.sample(size)
        x = np.random.uniform(0, q)

        self.x_kde = stats.kde.gaussian_kde(x)

    @classmethod
    def sample(self, N, validate=True):
        x = self.x_kde.resample(N)[0]
        i = np.random.uniform(-1, 1, N)

        if validate:
            valid = (x > 0) & (x < 1)
            x = x[valid]
            i = i[valid]
            N = len(x)
        
        q = np.sqrt(i**2 * (1 - x**2) + x**2)

        return q, x, i
    
    @classmethod
    def generate_posterior_x_kde(self, N, bw_method):
        # Generate valid samples
        q, x, i = self.sample(N)

        # Calculate sample weights
        q_kde = sm.nonparametric.KDEUnivariate(q)
        q_kde.fit(bw=0.03)
        q_pdf = q_kde.evaluate(self.q_pdf.x)
        weights = self.q_pdf.interp(q) / np.interp(q, self.q_pdf.x, q_pdf)

        # Create posterior xz distribution
        posterior = random.choices(np.indices(weights.shape)[0], weights, k=N)
        
        self.x_kde = stats.kde.gaussian_kde(x[posterior], bw_method)

    @classmethod
    def run(self, methods=[(150000, 0.05)]*50):
        for method in methods:
            self.generate_posterior_x_kde(*method)

        # Generate qt and qp kde
        q, x, i = self.sample(method[0])

        self.qi_kde = stats.kde.gaussian_kde(
            np.column_stack((q, i)).T, 0.05
        )
    
    @classmethod
    def get_i_pdf(self, q, values=100):
        x = np.linspace(0, 1, values)
        y = self.qi_kde(np.column_stack(
            (np.repeat(q, values), x
        )).T)

        return PDF(x, y)
