import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def get_best_estimator(Estimator, bounds, target, epsilon, density=5, max_iterations=10):
    if isinstance(density, int):
        density = np.repeat(density, len(bounds))

    x = [np.linspace(b[0], b[1], density[i]) for i, b in enumerate(bounds)]
    grid = np.array(np.meshgrid(*x)).T.reshape(-1, len(bounds))
    
    best_x = None
    best_estimator = None
    
    for x in grid:
        estimator = Estimator(x)
        estimator.error = estimator.fit(target)
        
        if best_estimator == None or estimator.error < best_estimator.error:
            best_x = x
            best_estimator = estimator
    
    if max_iterations > 0:
        done = True
        new_bounds = []

        for i, x in enumerate(best_x):
            step = (bounds[i][1] - bounds[i][0])/density[i]
            
            if step > epsilon[i]:
                done = False
            else:
                density[i] = 1

            new_bounds.append([
                max(bounds[i][0], x - step),
                min(bounds[i][1], x + step)
            ])

        if not done:
            return get_best_estimator(Estimator, new_bounds, target, epsilon, density, max_iterations - 1)
    
    return best_estimator


def get_truncnorm_sample(mu, sigma, a, b, N):
    return truncnorm.rvs((a - mu)/sigma, (b - mu)/sigma, loc=mu, scale=sigma, size=N)


def get_truncnorm_pdf(x, mu, sigma, a, b):
    return truncnorm.pdf(x, (a - mu)/sigma, (b - mu)/sigma, loc=mu, scale=sigma)


def plot_truncnorm_pdf(mu, sigma, a, b, **kwargs):
    x = np.linspace(a, b, 100)
    a_norm, b_norm = (a - mu) / sigma, (b - mu) / sigma
    rv = truncnorm(a_norm, b_norm)
    x_norm = np.linspace(a_norm, b_norm, 100)
    plt.plot(x, rv.pdf(x_norm), **kwargs)


def sample_x_slot(slot, N=1):
    return np.random.uniform(slot * 0.005, (slot + 1) * 0.005, N)


def sample_z_slot(slot, N=1):
    return np.random.uniform(0.7 + slot * 0.005, 0.7 + (slot + 1) * 0.005, N)


def sample_ba_hist(hist, N=1):
    #bin_midpoints = bins[:-1] + np.diff(bins)/2

    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(N)
    slots = np.searchsorted(cdf, values)

    return np.array([
        np.random.uniform(slot * 0.01, (slot + 1) * 0.01) for slot in slots
    ])


def sample_inclination(pdf, a=0, b=1, N=1):
    pdf = pdf[int(a*100):int(b*100)]

    if len(pdf) <= 1:
        return np.random.uniform(a, b, N)

    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    values = np.random.rand(N)
    slots = np.searchsorted(cdf, values)

    return a + np.array([
        np.random.uniform(slot * 0.01, (slot + 1) * 0.01) for slot in slots
    ])


class PDF:
    def __init__(self, pdf, grid, add_uniform=False):
        self.pdf = pdf
        self.cdf = np.cumsum(pdf)
        self.cdf = self.cdf / self.cdf[-1]
        self.grid = grid

        if add_uniform:
            self.add_uniform = self.grid[1] - self.grid[0]
        else:
            self.add_uniform = 0

    def sample(self, size):
        values = np.random.rand(size)
        value_bins = np.searchsorted(self.cdf, values)
        random_from_cdf = self.grid[value_bins]

        if self.add_uniform:
            return random_from_cdf + np.random.uniform(0, self.add_uniform, size)

        return random_from_cdf
    
    def plot(self, **kwargs):
        plt.plot(self.grid, self.pdf, **kwargs)
    
    def interp(self, x):
        return np.interp(x, self.grid, self.pdf)

