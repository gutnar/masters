#%%
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import random

from analytical import get_q
from lib.pdf import PDF


class BayesianApproximation:
    Z_MIN = 0.6

    @classmethod
    def __init__(self, q_pdf, size=10000, z_mu=0.85, z_sigma=0.1):
        self.q_pdf = q_pdf

        # Initialize xz kde
        q = q_pdf.sample(size)
        x = np.random.uniform(0, q)
        z = np.random.normal(z_mu, z_sigma, size)
        self.xz_kde = stats.kde.gaussian_kde(np.column_stack((x, z)).T)

    @classmethod
    def sample(self, N):
        x, z = self.xz_kde.resample(N)
        valid = (x > 0) & (x < z) & (z > self.Z_MIN) & (z < 1)

        x = x[valid]
        z = z[valid]
        N = len(x)
        
        p = np.random.uniform(0, np.pi, N)
        t = np.arccos(np.random.uniform(0, 1, N))
        q = get_q(x, z, p, t)

        return q, x, z, p, t
    
    @classmethod
    def generate_posterior_xz_kde(self, N, bw_method):
        # Generate valid xz samples
        q, x, z, p, t = self.sample(N)

        # Calculate sample weights
        q_kde = sm.nonparametric.KDEUnivariate(q)
        q_kde.fit(bw=0.03)
        q_pdf = q_kde.evaluate(self.q_pdf.x)
        weights = self.q_pdf.interp(q) / np.interp(q, self.q_pdf.x, q_pdf)

        # Create posterior xz distribution
        posterior = random.choices(np.indices(weights.shape)[0], weights, k=N)

        xz_posterior = np.column_stack((
            x[posterior], z[posterior]
        ))

        self.xz_kde = stats.kde.gaussian_kde(xz_posterior.T, bw_method)

    @classmethod
    def run(self, methods=[(150000, 0.05)]*50):
        for method in methods:
            self.generate_posterior_xz_kde(*method)

        # Generate qt and qp kde
        q, x, z, p, t = self.sample(method[0])

        self.qt_kde = stats.kde.gaussian_kde(
            np.column_stack((q, t)).T, 0.05
        )

        self.qp_kde = stats.kde.gaussian_kde(
            np.column_stack((q, p)).T, 0.05
        )
    
    @classmethod
    def get_t_pdf(self, q, values=100):
        return PDF(
            np.linspace(0, np.pi/2, values),
            self.qt_kde(np.column_stack(
                (np.repeat(q, values),
                np.linspace(0, np.pi/2, values)
            )).T)
        )
    
    @classmethod
    def get_p_pdf(self, q, values=100):
        return PDF(
            np.linspace(0, np.pi, values),
            self.qp_kde(np.column_stack(
                (np.repeat(q, values),
                np.linspace(0, np.pi, values)
            )).T)
        )
