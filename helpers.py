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
