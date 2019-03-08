#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
from scipy.optimize import fsolve, differential_evolution, brute
from helpers import get_truncnorm_sample, plot_truncnorm_pdf
from time import time
from multiprocessing import Pool
from os import cpu_count
import statsmodels.api as sm

#%%
plot_truncnorm_pdf(0.5, 0.25, 0, 1)
plot_truncnorm_pdf(0.5, 0.25, 0, 0.1)

#%%
from classifier import clf, parameters
from inclination import estimate_inclination

galaxies = pd.read_csv("data_inclinations.txt")

#%%
sorted_galaxies = galaxies.sort_values(by=["x_mu"], ascending=True)
galaxy = sorted_galaxies.iloc[[50]]

pdf = clf.predict_proba(galaxy[parameters].values)[0] / 0.01
result = estimate_inclination(pdf, True)

galaxy, result

#%%
from inclination import sample_cos_t
from scipy.stats import truncnorm

galaxy = galaxies.iloc[[9000]]

cos_t = sample_cos_t(
    float(galaxy["ba"]),
    float(galaxy["x_mu"]),
    float(galaxy["x_sigma"]),
    float(galaxy["z_mu"]),
    float(galaxy["z_sigma"]),
    10000
)

plt.hist(cos_t, 100, (0, 1))

#%%
from inclination import get_ba
from symbolic import test_cos_t

if __name__ == '__main__':
    N = 1000
    x = np.random.normal(0.3, 0.1, N)
    z = get_truncnorm_sample(0.9, 0.05, 0, 1, N)
    ba = get_ba(x, z)
    
    #cos_t = np.sqrt((ba**2 - x**2) / (1 - x**2))
    p = np.random.uniform(0, 2*np.pi, N)

    cos_t = [
        fsolve(test_cos_t, 0.5, (p[i], x[i], z[i], ba[i]))[0]
        for i in range(N)
    ]

    plt.hist(cos_t, 100)

#%%
from inclination import sample_cos_t

if __name__ == '__main__':
    galaxies = pd.read_csv("data_inclinations.txt")

    sample = galaxies
    print(sample.describe())

    #plt.figure(1)
    #plt.hist(galaxies["ba"], 100)

    #processes = cpu_count() - 1
    #chunks = np.array_split(sample, processes)

    start = time()
    #pool = Pool(processes)
    #angles = pd.concat(pool.map(process_galaxies, chunks))

    cos_t_samples = np.array([])

    for i in range(len(sample)):
        galaxy = sample.iloc[[i]]
        
        cos_t_samples = np.concatenate([
            cos_t_samples,
            sample_cos_t(
                float(galaxy["ba"]),
                float(galaxy["x_mu"]),
                float(galaxy["x_sigma"]),
                float(galaxy["z_mu"]),
                float(galaxy["z_sigma"]),
                10
            )
        ])
    
    print(time() - start)
