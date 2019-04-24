#%%
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import random
import math

from lib.pdf import PDF
from lib.binney import get_q, get_psi


class BayesianApproximation2d:
    Z_MIN = 0

    @classmethod
    def __init__(self, q_pdf, size=10000, zeta_mu=0.85, zeta_sigma=0.1):
        self.q_pdf = q_pdf

        # Initialize xz kde
        q = q_pdf.sample(size)
        xi = np.random.uniform(0, q)
        zeta = np.random.normal(zeta_mu, zeta_sigma, size)

        self.xz_kde = stats.kde.gaussian_kde(np.column_stack((xi, zeta)).T)

    @classmethod
    def sample(self, N, validate=True):
        xi, zeta = self.xz_kde.resample(N)

        if validate:
            valid = (xi > 0) & (xi < zeta) & (zeta > self.Z_MIN) & (zeta < 1)
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
    def run(self, methods=[(150000, 0.05)]*50):
        for method in methods:
            self.generate_posterior_xz_kde(*method)
    
    '''
    @classmethod
    def sample_spin_vec(self, q, p, N):
        q_sample, xi, zeta, theta, phi = self.sample(1000000)
        sample = (q_sample > (q - 0.01)) & (q_sample < (q + 0.01))

        kde = stats.kde.gaussian_kde(np.column_stack((
            xi[sample], zeta[sample], theta[sample], phi[sample]
        )).T)

        xi, zeta, theta, phi = kde.resample(N)

        return get_z_prime_vec(xi, zeta, theta, phi, p).T[:,0,:]
    '''
    
    @classmethod
    def sample_pos_inc(self, q, p, N):
        q_sample, xi, zeta, theta, phi = self.sample(1000000)
        sample = (q_sample > (q - 0.01)) & (q_sample < (q + 0.01))
        
        kde = stats.kde.gaussian_kde(np.column_stack((
            xi[sample], zeta[sample], theta[sample], phi[sample]
        )).T)

        xi, zeta, theta, phi = kde.resample(N)

        return (
            p - get_psi(xi, zeta, theta, phi),
            np.cos(theta)
        )
