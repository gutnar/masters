#%%
from helpers import get_truncnorm_sample, get_truncnorm_pdf, plot_truncnorm_pdf, sample_ba_hist, sample_inclination
from scipy import stats
from time import time
from scipy.optimize import differential_evolution
from sympy import Symbol, lambdify, sqrt, Abs, sin, cos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


p = Symbol("p", positive=True)
x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)

expression = sqrt(q**2*x**2*z**2*sin(p)**2/2 - q**2*x**2*z**2/2 - q**2*x**2*sin(p)**2/2 + q**2*z**2/2 + x**4 - x**2*z**2*sin(p)**2 + x**2*sin(p)**2 - x**2 + z**4*sin(p)**4 - z**4*sin(p)**2 - 2*z**2*sin(p)**4 + 2*z**2*sin(p)**2 - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/2 + sin(p)**4 - sin(p)**2 + x**2*z**2*sin(p)**2/(2*q**2) - x**2*z**2/(2*q**2) - x**2*sin(p)**2/(2*q**2) + z**2/(2*q**2) - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/(2*q**2))/Abs(x**2 - z**2*sin(p)**2 + sin(p)**2 - 1)
cos_t = lambdify([p, x, z, q], expression, "numpy")



def test_parameters(i, ba_samples, p, N):
    cos_t_samples = []

    for ba in ba_samples:
        cos_t_samples.append(cos_t(
            np.random.uniform(0, 2*np.pi),
            get_truncnorm_sample(i[0], i[1], 0, ba, 1)[0],
            get_truncnorm_sample(i[2], i[3], ba, 1, 1)[0],
            ba
        ))

    #cos_t_samples = cos_t(p, x, z, ba)
    cos_t_samples = np.array(cos_t_samples)
    cos_t_samples = cos_t_samples[~np.isnan(cos_t_samples)]
    hist = np.histogram(cos_t_samples, 100, (0, 1), density=True)[0]

    return np.sum(np.square((
        hist - np.mean(hist)
    )))


def estimate_inclination(hist, plot=False, plot_color=None, plot_label=None):
    N = 200
    ba = sample_ba_hist(hist, N)
    p = np.random.uniform(0, 2*np.pi, N)

    result = differential_evolution(test_parameters, (
        (0.01, 0.5), (0, 0.5), (0.5, 0.9999), (0, 0.25)
    ), args=(ba, p, N), maxiter=1).x

    if plot:
        N = 10000
        ba = sample_ba_hist(hist, N)
        p = np.random.uniform(0, 2*np.pi, N)
        x = get_truncnorm_sample(result[0], result[1], 0, 1, N)
        z = get_truncnorm_sample(result[2], result[3], 0, 1, N)

        cos_t_samples = cos_t(p, x, z, ba)
        cos_t_samples = cos_t_samples[~np.isnan(cos_t_samples)]

        plt.hist(cos_t_samples, 100, (0, 1), histtype="step", color=plot_color, label=(
            "x ~ N(%.2f, %.2f), z ~ N(%.2f, %.2f)" %  
            (result[0], result[1], result[2], result[3])
        ))

    return result


def plot_quantile_inclination_results(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        plt.figure(1)

        ba = galaxies[quantiles == i]["ba"].values
        hist = np.histogram(ba, 100, (0, 1), density=True)[0]
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

#%%
if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [16, 5]
    galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", sep=r"\s+")

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

#%%
def bayesian(ba_pdf, x_pdf, z_pdf, N, i=1):
    slots = np.linspace(0, 1, 100)

    ba_samples = sample_ba_hist(ba_pdf, N)
    #p_samples = np.random.uniform(0, 2*np.pi, N)
    x_samples = np.array([])
    z_samples = np.array([])

    for ba in ba_samples:
        x_samples = np.concatenate((
            x_samples, sample_inclination(x_pdf, 0, ba, 1)
        ), axis=0)
        z_samples = np.concatenate((
            z_samples, sample_inclination(z_pdf, ba, 1, 1)
        ), axis=0)
    
    #x_samples = sample_ba_hist(x_pdf, N)
    #z_samples = sample_ba_hist(z_pdf, N)

    cos_t_samples = cos_t(np.pi, x_samples, z_samples, ba_samples)

    #keep = (x_samples < z_samples)
    #keep = keep & (x_samples < ba_samples)
    #keep = keep & (z_samples > ba_samples)
    #keep = keep & ~np.isnan(cos_t_samples)
    keep = ~np.isnan(cos_t_samples)
    
    ba_samples = ba_samples[keep]
    #p_samples = p_samples[keep]
    x_samples = x_samples[keep]
    z_samples = z_samples[keep]
    cos_t_samples = cos_t_samples[keep]

    cos_t_kde = sm.nonparametric.KDEUnivariate(cos_t_samples)
    cos_t_kde.fit(bw=0.05)
    
    weights = cos_t_kde.evaluate(cos_t_samples)
    weights = 1 - (weights - np.min(weights)) / np.max(weights)

    posterior = np.random.rand(len(cos_t_samples)) < weights
    cos_t_kde = sm.nonparametric.KDEUnivariate(cos_t_samples[posterior])
    cos_t_kde.fit(bw=0.05)

    x_kde = sm.nonparametric.KDEUnivariate(x_samples[posterior])
    x_kde.fit()
    z_kde = sm.nonparametric.KDEUnivariate(z_samples[posterior])
    z_kde.fit()

    if i == 0:
        plt.figure(1)
        plt.xlim((0, 1))
        plt.plot(cos_t_kde.support, cos_t_kde.density, label="cos(t)")
        plt.gca().legend()
        
        plt.figure(2)
        plt.xlim((0, 1))
        plt.plot(x_kde.support, x_kde.density, label="x")
        plt.plot(z_kde.support, z_kde.density, label="z")
        plt.gca().legend()

    if i > 0:
        return bayesian(
            ba_pdf,
            x_kde.evaluate(slots),
            z_kde.evaluate(slots),
            N,
            i - 1
        )

if __name__ == "__main__":
    ba_hist = np.histogram(galaxies["ba"].values, 100, (0, 1), density=True)[0]
    x_pdf = get_truncnorm_pdf(np.linspace(0, 1, 100), 0.3, 0.1, 0, 1)
    z_pdf = get_truncnorm_pdf(np.linspace(0, 1, 100), 0.9, 0.05, 0, 1)

    bayesian(ba_hist, x_pdf, z_pdf, 1000, 10)


#%%
ba_pdf = np.histogram(galaxies["ba"].values, 100, (0, 1), density=True)[0]
ba = sample_ba_hist(ba_pdf, 10000)
ba_hist = np.histogram(ba, 100, (0, 1))[0]

kde = sm.nonparametric.KDEUnivariate(ba)
kde.fit(bw=0.05)

plt.hist(ba, 100, (0, 1), histtype="step", density=True)
plt.plot(kde.support, kde.density)

#%%
weights = kde.evaluate(ba)
weights = (weights - np.min(weights)) / np.max(weights)
plt.hist(weights, 100, (0, 1))

#%%
posterior = np.greater(np.random.rand(len(ba)), weights)
plt.hist(ba[posterior], 100, (0, 1), density=True, histtype="step")

sum(posterior)
