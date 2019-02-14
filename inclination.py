#%%
from helpers import get_truncnorm_sample
from scipy import stats
from time import time
from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


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


def test_parameters(parameters, target, N):
    x = get_truncnorm_sample(parameters[0], parameters[1], 0, 1, N)
    z = get_truncnorm_sample(parameters[2], parameters[3], 0, 1, N)
    ba = get_ba(x, z)

    kde = sm.nonparametric.KDEUnivariate(ba)
    kde.fit()
    ba_hist = kde.evaluate(np.linspace(0, 1, 100) + 0.005)

    return np.sum(np.abs((target - ba_hist)))


def estimate_inclination(hist, plot=False, plot_color=None, plot_label=None):
    result = differential_evolution(test_parameters, (
        (0.01, 0.4), (0, 0.25), (0.85, 0.99), (0, 0.1)
    ), args=(hist, 1000), maxiter=50).x

    if plot:
        x = get_truncnorm_sample(result[0], result[1], 0, 1, 10000)
        z = get_truncnorm_sample(result[2], result[3], 0, 1, 10000)
        ba = get_ba(x, z)
        kde = stats.gaussian_kde(ba)
        
        bins = np.linspace(0, 1, 100)
        plt.plot(bins, hist, 'o', color=plot_color, label=plot_label)
        plt.plot(bins, kde(bins + 0.005), color=plot_color, label=(
            "x ~ N(%.2f, %.2f), z ~ N(%.2f, %.2f)" %  
            (result[0], result[1], result[2], result[3])
        ))

    return result


def plot_quantile_inclination_results(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        hist = np.histogram(galaxies[quantiles == i]["ba"].values, 100, (0, 1), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']

        start = time()
        result = estimate_inclination(hist, True, color, "("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"] "+parameter)
        print(time() - start, result)

        plt.title(parameter)
        plt.gca().legend()
        plt.savefig("plots/inclination_" + parameter + "_hist.png")


#%%
if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [16, 5]
    galaxies = pd.read_csv("data_gama_gal_orient.txt", sep=r"\s+")

#%%
if __name__ == '__main__':
    plot_quantile_inclination_results(galaxies, "rmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    plot_quantile_inclination_results(galaxies, "rabsmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    plot_quantile_inclination_results(galaxies, "redshift", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    plot_quantile_inclination_results(galaxies, "rad", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    plot_quantile_inclination_results(galaxies, "sern", (0, 1/10, 1))
