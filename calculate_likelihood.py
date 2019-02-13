#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import get_ba_hist, get_truncnorm_sample

#%%
if __name__ == '__main__':
    galaxies = pd.read_csv("data_inclinations.5000.txt")
    
    for i in range(1):
        galaxy = galaxies.iloc[[i]]
        x = get_truncnorm_sample(galaxy["x_mu"], galaxy["x_sigma"], 0, 1, 10000)
        z = get_truncnorm_sample(galaxy["z_mu"], galaxy["z_sigma"], 0, 1, 10000)

        ba_hist = get_ba_hist(x, z)
        plt.plot(ba_hist)
