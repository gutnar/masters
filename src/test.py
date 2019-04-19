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
galaxies = pd.concat((spiral, elliptic))

plt.hist(galaxies["sern"], 100, histtype="step")

galaxies.describe()

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

galaxies = pd.read_csv("data/intermediate/spiral.csv")

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
    np.arccos(galaxies["ba"]),
    galaxies["gama"],
    galaxies["ex"],
    galaxies["ey"],
    galaxies["ez"]
)), 100, density=True, histtype="step")
