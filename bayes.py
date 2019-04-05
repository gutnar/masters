#%%
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import warnings
from sklearn.preprocessing import normalize
from random import choices
from time import time

from helpers import PDF
from analytical import get_q, get_p_domain, get_cos_t, get_dum

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    plt.rcParams.update({
        "figure.figsize": [16, 10],
        "font.size": 16
    })

def plot_kde(kde, xlabel, ylabel, resample=True):
    if resample:
        x, z = kde.resample(10000)
        kde = stats.kde.gaussian_kde(np.column_stack((x, z)).T)
    
    pdf = kde(kde_grid.T).reshape(100, 100)
    
    plt.xlabel(xlabel)
    plt.xticks([0, 19, 39, 59, 79, 99], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.ylabel(ylabel)
    plt.yticks([0, 19, 39, 59, 79, 99], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.imshow(pdf, "magma", origin="lower")

def plot_xz_kde(kde, resample=True):
    plot_kde(kde, "x", "z", resample)

def plot_qt_kde(kde, resample=True):
    plot_kde(kde, "q", "cos(t)", resample)

#%%
q_grid = np.linspace(0, 1, 100)

kde_mesh = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100)
)

kde_grid = np.append(kde_mesh[0].reshape(-1, 1), kde_mesh[1].reshape(-1, 1), axis=1)

#%%
def sample_qxzpt(xz_kde, N):
    x, z = xz_kde.resample(N)
    #valid = (x > 0) & (x < z) & (z < 1)
    valid = (x > 0) & (x < z) & (z > 0.5) & (z < 1)
    x = x[valid]
    z = z[valid]
    N = len(x)
    
    p = np.random.uniform(0, np.pi, N)
    t = np.arccos(np.random.uniform(0, 1, N))
    q = get_q(x, z, p, t)

    return q, x, z, p, t


def get_xz_posterior_kde(q_target_pdf, xz_kde, N, bw_method):
    # Generate valid xz samples
    q, x, z, p, t = sample_qxzpt(xz_kde, N)

    # Calculate sample weights
    q_sample_kde = sm.nonparametric.KDEUnivariate(q)
    q_sample_kde.fit(bw=0.03)
    q_sample_pdf = PDF(q_sample_kde.evaluate(q_grid), q_grid)

    weights = q_target_pdf.interp(q) / q_sample_pdf.interp(q)

    # Create posterior xz distribution
    #posterior = choices(np.indices(weights.shape)[0], weights, k=N)
    #x = x[posterior]
    #z = z[posterior]

    #cos_t_posterior = np.cos(t[posterior])
    #cos_t_posterior_kde = sm.nonparametric.KDEUnivariate(cos_t_posterior)
    #cos_t_posterior_kde.fit(bw=0.03)
    #cos_t_posterior_pdf = PDF(cos_t_posterior_kde.evaluate(q_grid), q_grid)

    #weights = np.mean(cos_t_posterior_pdf.pdf) / cos_t_posterior_pdf.interp(cos_t_posterior)

    # Create posterior xz distribution
    posterior = choices(np.indices(weights.shape)[0], weights, k=N)

    xz_posterior = np.column_stack((
        x[posterior], z[posterior]
    ))

    return stats.kde.gaussian_kde(xz_posterior.T, bw_method)


def get_xz_kde(q_pdf, plot=False, methods=[(1000, "scott")]*100 + [(150000, 0.05)]*50):
    q_samples = q_pdf.sample(10000)
    z = np.random.normal(0.85, 0.1, 10000)
    x = np.random.uniform(0, q_samples)
    xz_kde = stats.kde.gaussian_kde(np.column_stack((x, z)).T)

    for method in methods:
        xz_kde = get_xz_posterior_kde(q_pdf, xz_kde, method[0], method[1])

    if plot:
        q, x, z, p, t = sample_qxzpt(xz_kde, 10000)
        q_sample_kde = sm.nonparametric.KDEUnivariate(q)
        q_sample_kde.fit(bw=0.03)
        
        plt.figure(1)
        plot_xz_kde(xz_kde)

        plt.figure(2)
        plt.xlim((0, 1))
        plt.plot(q_pdf.grid, q_pdf.pdf, label="target q")
        plt.plot(q_sample_kde.support, q_sample_kde.density, label="q")
        plt.hist(np.cos(t), 100, (0, 1), True, label=r"$\cos(\theta)$", histtype="step")
        plt.hist(p/np.pi, 100, (0, 1), True, label=r"$\phi/\pi$", histtype="step")
        plt.gca().legend()

        #plt.figure(3)
        #qt_kde = stats.kde.gaussian_kde(np.column_stack((q, np.cos(t))).T)
        #plot_qt_kde(qt_kde, False)

    return xz_kde


def sample_cos_t(q, xz_kde, N):
    x, z = xz_kde.resample(N)
    valid = (x > 0) & (x < q) & (z > q) & (z < 1)
    x = x[valid]
    z = z[valid]
    N = len(x)
    
    p_min, p_max = get_p_domain(q, x, z)
    tf = np.array([0, 1])
    p = np.random.uniform(p_min, p_max) + np.random.choice(tf, N) * np.pi

    cos_t = get_cos_t(q, x, z, p)

    cos_t[x > q] = 0
    cos_t[z < q] = 1

    return cos_t

#%%
if __name__ == "__main__":
    #gama = pd.read_csv("data/raw/gama_data_for_gutnar.txt", r"\s+")
    #inclinations = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
    #galaxies = gama.merge(inclinations, on="id", how="inner")

    galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
    #galaxies = galaxies[galaxies["ba"] > 0.1]

    parameter = "rabsmag"
    cuts = (0, 1/10, 1)
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)
    #galaxies = galaxies[quantiles == 1]

    q_kde = sm.nonparametric.KDEUnivariate(galaxies["ba"].values)
    q_kde.fit(bw=0.01)
    q_grid = np.linspace(0, 1, 100, False)
    q_pdf = PDF(q_kde.evaluate(q_grid), q_grid, True)

    plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
    q_pdf.plot()

#%%
if __name__ == "__main__":
    start = time()
    xz_kde = get_xz_kde(q_pdf, True, [(1000, "scott")]*100 + [(150000, 0.05)]*50)
    print(time() - start)

#%%
if __name__ == "__main__":
    q, x, z, p, t = sample_qxzpt(xz_kde, 150000)

    qt_kde = stats.kde.gaussian_kde(np.column_stack((q, np.cos(t))).T, 0.05)
    qp_kde = stats.kde.gaussian_kde(np.column_stack((q, p/np.pi)).T, 0.05)
    
    plt.figure(1)
    plot_kde(qt_kde, r"$q$", r"$cos(\theta)$")
    
    plt.figure(2)
    plot_kde(qp_kde, r"$q$", r"$\phi/\pi$")

    #plt.figure(3)
    #q_pdf.plot()
    #plt.hist(q, 100, (0, 1), True, histtype="step", label=r"$q$")
    #plt.hist(cos_t, 100, (0, 1), True, histtype="step", label=r"$\cos(\theta)$")
    #plt.hist(p/np.pi, 100, (0, 1), True, histtype="step", label=r"$\phi/\pi$")
    #plt.gca().legend()


#%%
if __name__ == "__main__":
    test_q = [0.01, 0.5, 0.99]

    for i, q in enumerate(test_q):
        start = time()
        
        test_cos_t_pdf = PDF(qt_kde(np.column_stack(
            (np.repeat(q, 100),
            np.linspace(0, 1, 100)
        )).T), np.linspace(0, 1, 100), True)
        
        test_p_pdf = PDF(qp_kde(np.column_stack(
            (np.repeat(q, 100),
            np.linspace(0, 1, 100)
        )).T), np.linspace(0, 1, 100), True)

        print(time() - start)
        
        plt.figure(i + 1)
        plt.title(r"$q = %.2f$" % q)
        test_cos_t_pdf.plot(label=r"$\cos(\theta)$")
        test_p_pdf.plot(label=r"$\phi/\pi$")
        plt.gca().legend()

#%%
if __name__ == "__main__":
    #q_samples, x, z, p, t = sample_qxzpt(xz_kde, 1000)
    q_samples = q_pdf.sample(1000)
    #qt_pdf = qt_kde(kde_grid.T).reshape(100, 100)
    cos_t_samples = np.array([])

    for q in q_samples:
        #cos_t_pdf = PDF(qt_pdf[:,int(q*100)], q_grid)
        cos_t_pdf = PDF(qt_kde(np.column_stack((np.repeat(q, 25), np.linspace(0, 1, 25))).T), np.linspace(0, 1, 25), True)
        cos_t = cos_t_pdf.sample(10)
        cos_t_samples = np.concatenate((cos_t_samples, cos_t))
    
    plt.hist(q_samples, 100, (0, 1), True, histtype="step")
    plt.hist(cos_t_samples, 100, (0, 1), True, histtype="step")

#%%
if __name__ == "__main__":
    gama = pd.read_csv("data/raw/gama_data_for_gutnar.txt", r"\s+")
    inclinations = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
    galaxies = gama.merge(inclinations, on="id", how="inner")

    dum = np.array([])
    dum_simple = np.array([])
    dum_random = np.array([])
    cos_t_samples = np.array([])
    index = 0

    for i, galaxy in galaxies.iterrows():
        cos_t = sample_cos_t(galaxy["ba"], xz_kde, 50)
        index += 1

        cos_t_samples = np.concatenate((cos_t_samples, cos_t))

        dum = np.concatenate((
            dum, get_dum(
                galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
                galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
            )
        ), axis=None)

        # Simple flatness estimation
        dum_simple = np.concatenate((
            dum_simple, get_dum(
                galaxy["ra"], galaxy["dec"], galaxy["pos"], (float(galaxy["ba"]), ),
                galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
            )
        ))

        # Random inclination angle
        cos_t = np.random.uniform(0, 1, 10)

        dum_random = np.concatenate((
            dum_random, get_dum(
                galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
                galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
            )
        ))
    
    # cos(t)
    kde = sm.nonparametric.KDEUnivariate(cos_t_samples)
    kde_simple = sm.nonparametric.KDEUnivariate(galaxies["ba"].values)

    kde.fit(bw=0.05, cut=0)
    kde_simple.fit(bw=0.05, cut=0)

    plt.figure(1)
    plt.plot(kde_simple.support, kde_simple.density, label="b/a")
    plt.plot(kde.support, kde.density, label="bayes")
    #plt.hist(galaxies["ba"].values, 100, (0, 1), True, histtype="step")
    #plt.hist(np.array(cos_t_samples).flatten(), 100, (0, 1), True, histtype="step")

    # DUM
    kde = sm.nonparametric.KDEUnivariate(dum)
    kde_simple = sm.nonparametric.KDEUnivariate(dum_simple)
    kde_random = sm.nonparametric.KDEUnivariate(dum_random)

    kde.fit(bw=0.05, cut=0)
    kde_simple.fit(bw=0.05, cut=0)
    kde_random.fit(bw=0.05, cut=0)

    plt.figure(2)
    plt.plot(kde_random.support, kde_random.density, label="random")
    plt.plot(kde_simple.support, kde_simple.density, label="b/a")
    plt.plot(kde.support, kde.density, label="bayes")
    plt.legend()

#%%
from classifier import predict_pdf

galaxy = galaxies.iloc[[666]]
galaxy_pdf = predict_pdf(galaxy)
galaxy_kde = get_xz_kde(galaxy_pdf, True, [(1000, "scott")]*100 + [(150000, 0.05)]*50)

#%%
test_pdf = PDF(np.histogram(np.concatenate((
    np.random.normal(0.25, 0.1, 100000),
    np.random.normal(0.8, 0.2, 100000)
)), 100, (0, 1), True)[0], np.linspace(0, 1, 100, False))

test_pdf.plot()

#%%
start = time()
get_xz_kde(test_pdf, True, [(1000, "scott")]*100 + [(100000, 0.025)]*50)
time() - start
