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

    cos_p = np.array([])

    for i in range(len(sample)):
        galaxy = sample.iloc[[i]]
        ba = float(galaxy["ba"])
        x = get_truncnorm_sample(galaxy["x_mu"], galaxy["x_sigma"], 0, ba, 100)
        cos_p = np.concatenate([
            cos_p,
            np.sqrt((ba**2 - x**2) / (1 - x**2))
        ])
    
    print(time() - start)

    plt.hist(cos_p, 100, (0, 1), density=True)
    
    kde = sm.nonparametric.KDEUnivariate(cos_p)
    kde.fit()
    plt.plot(kde.support, kde.density)

    #angles.to_csv("angles.txt", index=False)
