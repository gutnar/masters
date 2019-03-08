#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from inclination import sample_cos_t
from time import time

#%%
def plot_cos_t(sample):
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
    
    cos_t_samples = cos_t_samples[~np.isnan(cos_t_samples)]

    plt.hist(cos_t_samples, 100, (0, 1), density=True)

    kde = sm.nonparametric.KDEUnivariate(cos_t_samples)
    kde.fit()
    plt.plot(kde.support, kde.density)


#%%
if __name__ == "__main__":
    galaxies = pd.read_csv("data/intermediate/inclinations.txt")

    start = time()
    plot_cos_t(galaxies)
    print(time() - start)