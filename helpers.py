import numpy as np
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


def get_ba(x, z):
    N = len(x)

    x2 = x**2
    z2 = z**2

    cos_t = np.random.uniform(0, 1, N)
    cos2_t = cos_t**2
    sin2_t = 1 - cos2_t

    cos_p = np.random.uniform(0, 1, N)
    cos2_p = cos_p**2
    sin2_p = 1 - cos2_p
    sin_2p = 2*np.sqrt(sin2_p)*np.sqrt(cos2_p)

    A = cos2_t/x2 * (sin2_p + cos2_p/z2) + sin2_t/z2
    B = (1 - 1/z2) * 1/x2 * cos_t * sin_2p
    C = (sin2_p/z2 + cos2_p)/x2
    D = np.sqrt((A - C)**2 + B**2)

    return np.sqrt((A + C - D) / (A + C + D))
