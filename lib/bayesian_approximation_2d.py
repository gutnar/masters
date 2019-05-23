#%%
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import random
import math

from lib.pdf import PDF
from lib.binney import get_q, get_psi


class BayesianApproximation2d:
    @classmethod
    def __init__(self, q_pdf, initial_kde=None):
        self.q_pdf = q_pdf

        # Initialize xz kde
        if initial_kde == None:
            zeta = np.random.normal(0.85, 0.1, 150000)
            xi = np.random.uniform(0, zeta)

            self.xz_kde = stats.kde.gaussian_kde(np.column_stack((xi, zeta)).T)
        else:
            self.xz_kde = initial_kde

    @classmethod
    def sample(self, N, validate=True):
        xi, zeta = self.xz_kde.resample(N)
        xi[xi < 0] = -xi[xi < 0]

        if validate:
            valid = (xi > 0) & (xi < zeta) & (zeta > 0.5) & (zeta < 1)
            xi = xi[valid]
            zeta = zeta[valid]
            N = len(xi)
        
        theta = np.concatenate((
            -np.arccos(np.random.uniform(-1, 1, math.floor(N/2))),
            np.arccos(np.random.uniform(-1, 1, math.ceil(N/2)))
        ))
        phi = np.random.uniform(-np.pi, np.pi, N)
        q = get_q(xi, zeta, theta, phi)

        return q, xi, zeta, theta, phi
    
    @classmethod
    def error(self, N):
        q, xi, zeta, theta, phi = self.sample(N)

        q_kde = sm.nonparametric.KDEUnivariate(q)
        q_kde.fit(bw=0.03)
        q_pdf = q_kde.evaluate(self.q_pdf.x)

        return np.sum((q_pdf - self.q_pdf.y)**2)
    
    @classmethod
    def generate_posterior_xz_kde(self, N, bw_method):
        # Generate valid xz samples
        q, xi, zeta, theta, phi = self.sample(N)

        # Calculate sample weights
        q_kde = sm.nonparametric.KDEUnivariate(q)
        q_kde.fit(bw=0.03)
        q_pdf = q_kde.evaluate(self.q_pdf.x)
        weights = self.q_pdf.interp(q) / np.interp(q, self.q_pdf.x, q_pdf)

        # Create posterior xz distribution
        posterior = random.choices(np.indices(weights.shape)[0], weights, k=N)

        xz_posterior = np.column_stack((
            xi[posterior], zeta[posterior]
        ))

        self.xz_kde = stats.kde.gaussian_kde(xz_posterior.T, bw_method)

    @classmethod
    def run(self, methods=[(150000, "scott")]*50):
        for method in methods:
            self.generate_posterior_xz_kde(*method)
    
    @classmethod
    def sample_pos_inc(self, q, p, N, bw=0.005):
        q_sample, xi, zeta, theta, phi = self.sample(1000000)
        sample = (q_sample > (q - bw)) & (q_sample < (q + bw))
        
        kde = stats.kde.gaussian_kde(np.column_stack((
            xi[sample], zeta[sample], theta[sample], phi[sample]
        )).T, 0.025)

        xi, zeta, theta, phi = kde.resample(N)
        #xi[xi < 0] = -xi[xi < 0]
        #zeta[zeta > 1] = 2 - zeta[zeta > 1]

        return (
            p - get_psi(xi, zeta, theta, phi),
            np.cos(theta)
        )
