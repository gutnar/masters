#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import get_truncnorm_pdf

#%%
galaxies = pd.read_csv("data_inclinations.txt")

#%%
galaxies.describe()

slots = np.linspace(0, 1, 100)
x_pdf = np.zeros(100)
z_pdf = np.zeros(100)

for i in range(len(galaxies)):
    galaxy = galaxies.iloc[[i]]
    x_pdf += get_truncnorm_pdf(slots, galaxy["x_mu"], galaxy["x_sigma"], 0, 1)
    z_pdf += get_truncnorm_pdf(slots, galaxy["z_mu"], galaxy["z_sigma"], 0, 1)

plt.plot(x_pdf)
plt.plot(z_pdf)
