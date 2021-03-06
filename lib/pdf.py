import numpy as np
import statsmodels.api as sm


class PDF:
    @staticmethod
    def from_samples(x, samples, **fit_args):
        kde = sm.nonparametric.KDEUnivariate(samples)
        kde.fit(**fit_args)

        return PDF(x, kde.evaluate(x))

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cdf = np.cumsum(y)
        self.cdf = self.cdf / self.cdf[-1]

    def sample(self, N):
        choices = np.random.rand(N)
        indices = np.searchsorted(self.cdf, choices)
        values = self.x[indices]
        
        return values
    
    def interp(self, x):
        return np.interp(x, self.x, self.y)
