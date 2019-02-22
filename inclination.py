#%%
from helpers import get_truncnorm_sample, plot_truncnorm_pdf
from scipy import stats
from time import time
from scipy.optimize import differential_evolution
from sympy import Symbol, lambdify, sqrt, Abs, sin, cos
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

    p = np.random.uniform(0, 2*np.pi, N)
    cos2_p = np.cos(p)**2
    sin2_p = np.sin(p)**2
    sin_2p = np.sin(2*p)

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

    return np.sum(np.square((target - ba_hist)))


def estimate_inclination(hist, plot=False, plot_color=None, plot_label=None):
    result = differential_evolution(test_parameters, (
        (0.01, 0.9999), (0, 0.5), (0.5, 0.9999), (0, 0.25)
    ), args=(hist, 1000), maxiter=25).x

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
        plt.figure(1)

        hist = np.histogram(galaxies[quantiles == i]["ba"].values, 100, (0, 1), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']

        start = time()
        result = estimate_inclination(hist, True, color, "("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"] "+parameter)
        print(time() - start, result)

        plt.figure(2)
        plot_truncnorm_pdf(result[0], result[1], 0, 1, color=color, label="x ~ N(%.2f, %.2f)" % (result[0], result[1]))
        plot_truncnorm_pdf(result[2], result[3], 0, 1, color=color, linestyle="--", label="z ~ N(%.2f, %.2f)" % (result[2], result[3]))

    plt.figure(1)
    plt.title(parameter)
    plt.gca().legend()
    plt.savefig("plots/inclination_" + parameter + "_hist.png")

    plt.figure(2)
    plt.gca().legend()


# Inverse
p = Symbol("p", positive=True)
x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)

expression = sqrt(q**2*x**2*z**2*sin(p)**2/2 - q**2*x**2*z**2/2 - q**2*x**2*sin(p)**2/2 + q**2*z**2/2 + x**4 - x**2*z**2*sin(p)**2 + x**2*sin(p)**2 - x**2 + z**4*sin(p)**4 - z**4*sin(p)**2 - 2*z**2*sin(p)**4 + 2*z**2*sin(p)**2 - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/2 + sin(p)**4 - sin(p)**2 + x**2*z**2*sin(p)**2/(2*q**2) - x**2*z**2/(2*q**2) - x**2*sin(p)**2/(2*q**2) + z**2/(2*q**2) - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/(2*q**2))/Abs(x**2 - z**2*sin(p)**2 + sin(p)**2 - 1)
cos_t = lambdify([p, x, z, q], expression, "numpy")

def sample_cos_t(ba, x_mu, x_sigma, z_mu, z_sigma, N):
    return cos_t(
        np.random.uniform(0, 2*np.pi, N),
        get_truncnorm_sample(x_mu, x_sigma, 0, ba, N),
        get_truncnorm_sample(z_mu, z_sigma, ba, 1, N),
        ba
    )

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
