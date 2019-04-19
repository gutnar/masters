#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib import get_dum

#%%
hist = pd.read_csv("data/final/filament_galaxies_random.csv")

hist.describe()

#%%
plt.imshow(hist.values.T, origin="lower")
plt.colorbar()
