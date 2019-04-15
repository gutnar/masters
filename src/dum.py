#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from analytical import get_dum

#%%
dum = {
    "simple": np.array([]),
    "random": np.array([]),
    "global": np.array([]),
    "classifier": np.array([])
}

#%% Filament galaxies
galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")

for i, galaxy in galaxies.iterrows():
    # Simple flatness estimation
    dum["simple"] = np.concatenate((
        dum["simple"], get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["pos"], (galaxy["ba"], ),
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ))

    # Random inclination angle
    dum["random"] = np.concatenate((
        dum["random"], get_dum(
            galaxy["ra"], galaxy["dec"], np.random.uniform(-90, 90, 10), np.random.uniform(0, 1, 10),
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ))

#%% Other methods
for method in ("global", "classifier"):
    galaxies = pd.read_csv("data/intermediate/filament_galaxies_%s.csv" % method)

    for i, galaxy in galaxies.iterrows():
        dum[method] = np.concatenate((
            dum[method], get_dum(
                galaxy["ra"], galaxy["dec"], galaxy["p"] * 180/np.pi, (np.cos(galaxy["t"]), ),
                galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
            )
        ), axis=None)

#%%
for method in ("global", "classifier"):
    galaxies = pd.read_csv("data/intermediate/filament_galaxies_%s.csv" % method)

    plt.hist(np.cos(galaxies["t"]), 100, (0, 1), True, histtype="step", label=method)

plt.legend()

#%%
plt.xlim((0, 1))
plt.ylim((0.85, 1.15))

for method in ("classifier",):
    kde = sm.nonparametric.KDEUnivariate(
        np.concatenate(
            (-1*dum[method], dum[method], 2 - dum[method])
        )
    )

    kde.fit(bw=0.03)
    plt.hist(dum[method], 100, (0, 1), True, histtype="step", label=method)
    #plt.plot(kde.support, kde.density*3, label=method)

plt.legend()

#plt.savefig("plots/dum.png")
