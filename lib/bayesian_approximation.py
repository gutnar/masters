#%%
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import random
import math

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
    def sample(self, N, validate=True):
        x, z = self.xz_kde.resample(N)

        if validate:
            valid = (x > 0) & (x < z) & (z > self.Z_MIN) & (z < 1)
            x = x[valid]
            z = z[valid]
            N = len(x)
        
        p = np.random.uniform(-np.pi, np.pi, N)
        t = np.concatenate((
            -np.arccos(np.random.uniform(-1, 1, math.floor(N/2))),
            np.arccos(np.random.uniform(-1, 1, math.ceil(N/2)))
        ))
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
        x = np.linspace(-np.pi/2, np.pi/2, values)
        y = self.qt_kde(np.column_stack(
            (np.repeat(q, values), x
        )).T)

        return PDF(x, (y + np.flip(y)) / y.sum() / 2)

    @classmethod
    def get_p_pdf(self, q, values=100):
        x = np.linspace(-np.pi/2, np.pi/2, values)
        y = self.qp_kde(np.column_stack(
            (np.repeat(q, values), x
        )).T)

        return PDF(x, (y + np.flip(y)) / y.sum() / 2)
    
    @classmethod
    def sample_tp(self, ba, N):
        q, x, z, p, t = self.sample(150000)
        valid = (q > (ba - 0.05)) & (q < (ba + 0.05))

        tp_kde = stats.kde.gaussian_kde(
            np.column_stack((t[valid], p[valid])).T, "scott"
        )

        return tp_kde.resample(N)
    
    @classmethod
    def sample_pos_inc(self, proj_q, proj_pos, N):
        #q, x, z, p, t = self.sample(N)
        x, z = self.xz_kde.resample(N)

        return (
            np.repeat(proj_pos, N),
            np.sqrt(np.maximum(0, np.minimum(1, (proj_q**2 - x**2) / (1 - x**2))))
        )

        '''
        valid = (q > (proj_q - 0.05)) & (q < (proj_q + 0.05))

        kde = stats.kde.gaussian_kde(
            np.column_stack((
                p[valid] + np.pi/2,
                #np.random.uniform(-np.pi/2, np.pi, sum(valid)),
                np.abs(np.cos(t[valid]))
            )).T, "scott"
        )

        pos, inc = kde.resample(N)
        inc = np.maximum(0, np.minimum(1, inc))

        return pos, inc
        '''
