#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Default plot size
#plt.rcParams['figure.figsize'] = [16, 5]

#%% Get galaxies and assign discrete ba slots
galaxies = pd.read_csv("data_gama_gal_orient.txt", sep=r"\s+")
galaxies["baslot"] = galaxies["ba"].multiply(100).apply(np.ceil) - 1

#%%
__name__ == '__main__' and galaxies.describe()

#%%
__name__ == '__main__' and galaxies["baslot"].hist(bins=100)

#%% Generate training and test sets
np.random.seed(0b11011100000011011011000001110000)
training_set = np.random.rand(len(galaxies)) < 0.75
galaxies_train = galaxies[training_set]
galaxies_test = galaxies[~training_set]

parameters = ["rmag", "rabsmag", "redshift", "rad", "sern"]

if __name__ == '__main__':
    print(len(galaxies_train), len(galaxies_test))
