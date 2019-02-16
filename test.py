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

inclinations = sum([
    [
        [0.01234958, 0.00923403, 0.97634919, 0.00111584],
        [0.01797638, 0.02303374, 0.97349781, 0.03035275],
        [0.03318095, 0.01287478, 0.98187451, 0.00750664]
    ], [
        [0.0167513 , 0.02986062, 0.98764488, 0.02340807],
        [0.02383499, 0.01394882, 0.97818459, 0.01015622], 
        [0.01595973, 0.00773543, 0.98498279, 0.00717259]
    ], [
        [0.02458909, 0.01763566, 0.98535531, 0.02357705],
        [0.01724567, 0.00417566, 0.98018954, 0.0741385 ]
    ], [
        [0.02012741, 0.00739975, 0.9752773 , 0.01528026],
        [0.02374853, 0.02678782, 0.98501453, 0.04491906]
    ]
], [])

sample = pd.DataFrame.from_records(inclinations, columns=("a", "b", "c", "d"))
inclinations = pd.DataFrame.from_records(inclinations, columns=("x_mu", "x_sigma", "z_mu", "z_sigma"))
pd.concat([sample, inclinations], axis=1, sort=False)

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
from symbolic import test_cos_t

if __name__ == '__main__':
    galaxies = pd.read_csv("data_inclinations.txt")

    sample = galaxies[:100]
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
        ba = float(galaxy["ba"])

        N = 1
        x = get_truncnorm_sample(galaxy["x_mu"], galaxy["x_sigma"], 0, ba, N)
        z = get_truncnorm_sample(galaxy["z_mu"], galaxy["z_sigma"], ba, 1, N)
        p = np.random.uniform(0, 2*np.pi, N)

        cos_t = np.array([
            #fsolve(test_cos_t, 0.5, (p[i], x[i], z[i], ba))[0]
            differential_evolution(test_cos_t, [
                (0, 1)
            ], args=(p[i], x[i], z[i], ba), maxiter=1).x[0]
            for i in range(N)
        ])
        
        cos_t_samples = np.concatenate([
            cos_t_samples,
            #np.sqrt((ba**2 - x**2) / (1 - x**2))
            cos_t
        ])
    
    print(time() - start)

    plt.hist(cos_t_samples, 100, (0, 1), density=True)
    
    kde = sm.nonparametric.KDEUnivariate(cos_t_samples)
    kde.fit()
    plt.plot(kde.support, kde.density)

    #angles.to_csv("angles.txt", index=False)

#%%
cos_t_samples = pd.read_csv("angles.txt")
#plt.hist(cos_t_samples, 100, (0, 1), density=True)

#kde = sm.nonparametric.KDEUnivariate(cos_t_samples)
#kde.fit()
#plt.plot(kde.support, kde.density)


#%%
values = cos_t_samples.where(
    (cos_t_samples != 0.0) & (cos_t_samples != 1.0)
).iloc[:, 0]

plt.hist(values, 100, (0, 1), density=True)

#kde = sm.nonparametric.KDEUnivariate(values)
#kde.fit()
#plt.plot(kde.support, kde.density)

#%%
plt.hist(np.random.uniform(0, 1, 1000), 100, (0, 1))