#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
from scipy.optimize import fsolve, differential_evolution, brute
from helpers import PDF
from time import time
from multiprocessing import Pool
from os import cpu_count
import statsmodels.api as sm
import scipy.stats as stats
from analytical import get_q, get_cos_t, get_p_domain
from helpers import PDF, get_truncnorm_sample
from random import choices

galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
parameter = "rabsmag"
cuts = (0, 1/10, 1)
quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)
galaxies = galaxies[quantiles == 0]

#%%
q_kde = sm.nonparametric.KDEUnivariate(galaxies["ba"].values)
q_kde.fit(bw=0.01)
q_grid = np.linspace(0, 1, 100)
q_pdf = PDF(q_kde.evaluate(q_grid), q_grid, 0.01)

plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
q_pdf.plot()

#%%
samples = q_pdf.sample(100000)
plt.hist(samples, 100, (0, 1), density=True)
q_pdf.plot()

#%%
q = np.random.uniform(0, 1, 10000)

qx = np.column_stack((
    q,
    get_truncnorm_sample(0.2, 0.1, 0, 1, 10000)
    #np.random.uniform(np.repeat(0, 10000), q)
))

qz = np.column_stack((
    q,
    get_truncnorm_sample(0.85, 0.1, 0, 1, 10000)
    #np.random.uniform(q, np.repeat(1, 10000))
))

x, y = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100)
)

grid = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

#%%
start = time()
qx_kde = stats.kde.gaussian_kde(qx.T)
qx_pdf = qx_kde(grid.T).reshape(100, 100)
print(time() - start)

plt.figure(1)
plt.xlabel("q")
plt.ylabel("x")
plt.imshow(qx_pdf, origin="lower")

start = time()
qz_kde = stats.kde.gaussian_kde(qz.T)
qz_pdf = qz_kde(grid.T).reshape(100, 100)
print(time() - start)

plt.figure(2)
plt.xlabel("q")
plt.ylabel("z")
plt.imshow(qz_pdf, origin="lower")

#%%
grid_1d = np.linspace(0, 1, 100)
grid_2d = grid

def bayesian(q_pdf, qx_kde, qz_kde, plot=False, N=10000):
    # Sample q from given q pdf
    q = q_pdf.sample(N)
    q = q[(q > 0) & (q < 1)]

    # Sample x from current qx pdf
    pdf = [
        PDF(qx_kde(np.column_stack((
            np.repeat(i/100, 25 + i),
            np.linspace(0, i/100, 25 + i)
        )).T), np.linspace(0, i/100, 25 + i)) for i in range(100)]

    x = np.array([pdf[int(q_sample*100)].sample(1)[0] for q_sample in q])
    
    # Sample z from current qz pdf
    pdf = [
        PDF(qz_kde(np.column_stack((
            np.repeat(i/100, 25 + (100 - i)),
            np.linspace(i/100, 1, 25 + (100 - i))
        )).T), np.linspace(i/100, 1, 25 + (100 - i))) for i in range(100)]
    
    z = np.array([pdf[int(q_sample*100)].sample(1)[0] for q_sample in q])

    # Sample phi
    p_min, p_max = get_p_domain(q, x, z)
    p = np.random.uniform(p_min, p_max)

    # Calculate cos(theta)
    cos_t = get_cos_t(q, x, z, p)

    # Remove invalid values
    valid = ~np.isnan(cos_t)
    q = q[valid]
    x = x[valid]
    z = z[valid]
    cos_t = cos_t[valid]

    # Test cos(t) distribution
    print("ks", stats.kstest(cos_t, "uniform"))

    # Find sample weights
    cos_t_kde = sm.nonparametric.KDEUnivariate(cos_t)
    cos_t_kde.fit(bw=0.03)
    cos_t_weights = 1/cos_t_kde.evaluate(cos_t)
    #weights = (weights - np.min(weights)) / np.max(weights)
    weights = cos_t_weights

    if plot:
        plt.figure(1)
        plt.xlim((0, 1))
        plt.hist(cos_t, 100, (0, 1), density=True)
        plt.plot(cos_t_kde.support, cos_t_kde.density)

    # Generate new pdfs
    posterior = choices(np.indices(weights.shape)[0], weights, k=N)
    #posterior = np.random.rand(len(weights)) > weights
    q = q[posterior]
    x = x[posterior]
    z = z[posterior]

    '''
    q_posterior = np.array([])
    x_posterior = np.array([])
    z_posterior = np.array([])

    for q_slot in range(100):
        q_min, q_max = q_slot / 100, (q_slot + 1)/100
        selection = (q >= q_min) & (q <= q_max)
        size = sum(selection)

        if size == 0:
            continue

        posterior = choices(np.indices((size, ))[0], weights[selection], k=100)
        
        q_values = np.random.uniform(q_slot/100, (q_slot + 1)/100, 100)
        x_values = x[selection][posterior]
        z_values = z[selection][posterior]

        q_posterior = np.append(q_posterior, q_values)
        x_posterior = np.append(x_posterior, x_values)
        z_posterior = np.append(z_posterior, z_values)
    '''

    qx_kde = stats.kde.gaussian_kde(np.column_stack((q, x)).T)
    qz_kde = stats.kde.gaussian_kde(np.column_stack((q, z)).T)

    if plot:
        plt.figure(1)
        plt.hist(cos_t[posterior], 100, (0, 1), density=True, histtype="step")

        plt.figure(2)
        pdf = qx_kde(grid_2d.T).reshape(100, 100)
        plt.xlabel("q")
        plt.ylabel("x")
        plt.imshow(pdf, origin="lower")

        plt.figure(3)
        pdf = qz_kde(grid_2d.T).reshape(100, 100)
        plt.xlabel("q")
        plt.ylabel("z")
        plt.imshow(pdf, origin="lower")

    return qx_kde, qz_kde

#%%
qx_kde_2, qz_kde_2 = bayesian(q_pdf, qx_kde, qz_kde, True)

#%%
start = time()
qx_kde_2, qz_kde_2 = bayesian(q_pdf, qx_kde_2, qz_kde_2, True)
time() - start

#%%
def get_qxz_pdf(q_pdf):
    pass

#%%
plt.hist(galaxies["ba"], 50, (0, 1), density=True, histtype="step")

#%%
data = np.random.normal(0.5, 0.25, 1000)
kde = sm.nonparametric.KDEUnivariate(data)
kde.fit(bw=0.03)

plt.figure(1)
plt.hist(data, 100, (0, 1), density=True)
plt.plot(kde.support, kde.density)

weights = 1/kde.evaluate(data)
data2 = choices(data, weights, k=10000)
plt.figure(2)
plt.hist(data2, 100, (0, 1), density=True)

#%%
x = get_truncnorm_sample(0.3, 0.1, 0, 1, 10000)
z = get_truncnorm_sample(0.9, 0.1, 0, 1, 10000)

xz = np.column_stack((x, z))

xz_meshes = [
    np.meshgrid(np.linspace(0, 0.4, 100, False), np.linspace(0.6, 1, 100))
]

xz_grid = np.append(
    np.vstack([mesh[0].reshape(-1, 1) for mesh in xz_meshes]),
    np.vstack([mesh[1].reshape(-1, 1) for mesh in xz_meshes]),
    axis=1
)

xz_mesh = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100)
)

xz_grid = np.append(xz_mesh[0].reshape(-1, 1), xz_mesh[1].reshape(-1, 1), axis=1)
#xz_grid = np.append(x_mesh, z_mesh, axis=1)

#%%
start = time()
xz_kde = stats.kde.gaussian_kde(xz.T)
xz_pdf = xz_kde(xz_grid.T).reshape(100, 100)
print(time() - start)

plt.figure(1)
plt.xlabel("x")
plt.ylabel("z")
plt.imshow(xz_pdf, origin="lower")

#%%
N = 5000
x, z = xz_kde.resample(N)
p = np.random.uniform(0, np.pi, N)
t = np.arccos(np.random.uniform(0, 1, N))

q_sample = get_q(x, z, p, t)
q_sample_kde = sm.nonparametric.KDEUnivariate(q_sample)
q_sample_kde.fit(bw=0.03)
q_sample_pdf = PDF(q_sample_kde.evaluate(q_grid), q_grid)

weights = q_pdf.interp(q_sample) / q_sample_pdf.interp(q_sample)
posterior = choices(np.indices(weights.shape)[0], weights, k=N)
#posterior = np.random.choice(np.indices(weights.shape)[0], N, p=weights)

q_posterior = q_sample[posterior]
q_posterior_kde = sm.nonparametric.KDEUnivariate(q_posterior)
q_posterior_kde.fit(bw=0.03)

xz_posterior = np.column_stack((
    x[posterior], z[posterior]
))

start = time()
xz_posterior_kde = stats.kde.gaussian_kde(xz_posterior.T)
xz_posterior_pdf = xz_posterior_kde(xz_grid.T).reshape(100, 100)
print(time() - start)

xz_kde = xz_posterior_kde

plt.figure(1)
plt.title("q")
plt.xlim((0, 1))
plt.plot(q_kde.support, q_kde.density, label="target q")
plt.plot(q_sample_kde.support, q_sample_kde.density, label="prior q")
plt.plot(q_posterior_kde.support, q_posterior_kde.density, label="posterior q")
plt.gca().legend()

plt.figure(2)
plt.title("Next iteration weights histogram")
plt.hist(weights, 100, (0, 2))

plt.figure(3)
plt.title("xz distribution")
plt.xlabel("x")
plt.ylabel("z")
plt.imshow(xz_posterior_pdf, origin="lower")

plt.figure(4)
plt.title("cos(t) histogram")
plt.hist(np.cos(t[posterior]), 100, (0, 1), density=True)

#%%
xz_integral = []

for q_slot in range(101):
    q_value = q_slot / 100
    xz_integral.append(xz_kde.integrate_box((0, q_value), (q_value, 1)))

plt.plot(np.linspace(0, 1, 101), xz_integral)

#%%
def bayesian2(q_pdf, xz_kde, plot=False, N=10000):
    # Sample q from given q pdf
    q = q_pdf.sample(N)
    q = q[(q > 0) & (q < 1)]

    # Sample phi
    p_min, p_max = get_p_domain(q, x, z)
    p = np.random.uniform(p_min, p_max)

    # Calculate cos(theta)
    cos_t = get_cos_t(q, x, z, p)

    # Remove invalid values
    valid = ~np.isnan(cos_t)
    q = q[valid]
    x = x[valid]
    z = z[valid]
    cos_t = cos_t[valid]

    # Test cos(t) distribution
    print("ks", stats.kstest(cos_t, "uniform"))

    # Find sample weights
    cos_t_kde = sm.nonparametric.KDEUnivariate(cos_t)
    cos_t_kde.fit(bw=0.03)
    cos_t_weights = 1/cos_t_kde.evaluate(cos_t)
    #weights = (weights - np.min(weights)) / np.max(weights)
    weights = cos_t_weights

    if plot:
        plt.figure(1)
        plt.xlim((0, 1))
        plt.hist(cos_t, 100, (0, 1), density=True)
        plt.plot(cos_t_kde.support, cos_t_kde.density)

    # Generate new pdfs
    posterior = choices(np.indices(weights.shape)[0], weights, k=N)
    #posterior = np.random.rand(len(weights)) > weights
    q = q[posterior]
    x = x[posterior]
    z = z[posterior]

    qx_kde = stats.kde.gaussian_kde(np.column_stack((q, x)).T)
    qz_kde = stats.kde.gaussian_kde(np.column_stack((q, z)).T)

    if plot:
        plt.figure(1)
        plt.hist(cos_t[posterior], 100, (0, 1), density=True, histtype="step")

        plt.figure(2)
        pdf = qx_kde(grid_2d.T).reshape(100, 100)
        plt.xlabel("q")
        plt.ylabel("x")
        plt.imshow(pdf, origin="lower")

        plt.figure(3)
        pdf = qz_kde(grid_2d.T).reshape(100, 100)
        plt.xlabel("q")
        plt.ylabel("z")
        plt.imshow(pdf, origin="lower")

    return qx_kde, qz_kde

