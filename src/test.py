#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib
import analytical

#%%
spiral = pd.read_csv("data/intermediate/spiral.csv")
elliptic = pd.read_csv("data/intermediate/elliptic.csv")

plt.hist(spiral["ba"], 100, (0, 1), True, histtype="step")
plt.hist(elliptic["ba"], 100, (0, 1), True, histtype="step")

#%%
plt.hist(np.concatenate(lib.get_dum(
    spiral["ra"],
    spiral["dec"],
    np.random.uniform(-np.pi/2, np.pi/2, len(spiral)),
    #spiral["pos"] / 180 * np.pi,
    np.arccos(spiral["ba"]),
    spiral["gama"],
    spiral["ex"],
    spiral["ey"],
    spiral["ez"]
)), 100, density=True, histtype="step")

plt.hist(np.concatenate(lib.get_dum(
    elliptic["ra"],
    elliptic["dec"],
    np.random.uniform(-np.pi/2, np.pi/2, len(elliptic)),
    #elliptic["pos"] / 180 * np.pi,
    np.arccos(elliptic["ba"]),
    elliptic["gama"],
    elliptic["ex"],
    elliptic["ey"],
    elliptic["ez"]
)), 100, density=True, histtype="step")

#%%
galaxies = pd.concat((spiral, elliptic))

plt.hist(galaxies["sern"], 100, histtype="step")

galaxies.describe()

galaxies.to_csv("data/intermediate/filament_galaxies.csv", index=False)

#%%
elliptic.describe()

#%%
spiral.describe()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lib
import analytical

galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")
galaxies = galaxies

dum = []

for i, galaxy in galaxies.iterrows():
    dum1, dum2 = analytical.get_dum(
        float(galaxy["ra"]),
        float(galaxy["dec"]),
        float(galaxy["pos"]),
        (float(galaxy["ba"]), ),
        int(galaxy["gama"]),
        float(galaxy["ex"]),
        float(galaxy["ey"]),
        float(galaxy["ez"])
    )

    dum.append(dum1)
    dum.append(dum2)

plt.hist(dum, 100, density=True, histtype="step")

plt.hist(np.concatenate(lib.get_dum(
    galaxies["ra"],
    galaxies["dec"],
    galaxies["pos"] / 180 * np.pi,
    galaxies["ba"],
    galaxies["gama"],
    galaxies["ex"],
    galaxies["ey"],
    galaxies["ez"]
)), 100, density=True, histtype="step")

#%%
values = np.array([])

for i in range(100):
    dum1, dum2 = lib.get_dum(
        galaxies["ra"],
        galaxies["dec"],
        np.random.uniform(0, np.pi),
        np.arccos(np.random.uniform(0, 1)),
        galaxies["gama"],
        galaxies["ex"],
        galaxies["ey"],
        galaxies["ez"]
    )

    values = np.concatenate((values, dum1, dum2))

#plt.ylim((0.95, 1.05))
plt.hist(values, 100, density=True, histtype="step")

#%%
import scipy.stats as stats

kde = stats.kde.gaussian_kde(np.column_stack((
    np.random.normal(0.2, 0.1, 100000),
    np.random.normal(0.85, 0.1, 100000),
    np.random.uniform(0, 1, 100000),
    np.random.uniform(0, 1, 100000)
)).T)

kde.resample(10000).T
