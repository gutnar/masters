#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
spiral = pd.read_csv("data/intermediate/spiral_galaxies.csv")
elliptic = pd.read_csv("data/intermediate/elliptic_galaxies.csv")

plt.hist(spiral["ba"], 100, (0, 1), True, histtype="step")
plt.hist(elliptic["ba"], 100, (0, 1), True, histtype="step")

#%%
elliptic.describe()

#%%
spiral.describe()
