#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from lib import get_dum

#%%
dum = {
    "simple": np.array([]),
    "random": np.array([]),
    "global": np.array([]),
    "classifier": np.array([])
}

cos = {
    "simple": np.array([]),
    "random": np.array([]),
    "global": np.array([]),
    "classifier": np.array([])
}

phi = {
    "simple": np.array([]),
    "random": np.array([]),
    "global": np.array([]),
    "classifier": np.array([])
}

#%% Filament galaxies
galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")
galaxies = galaxies.loc[galaxies.index.repeat(10)]

# Simple flatness estimation
cos["simple"] = galaxies["ba"]
phi["simple"] = galaxies["pos"]/180*np.pi
dum["simple"] = np.concatenate(get_dum(
    galaxies["ra"], galaxies["dec"],
    galaxies["pos"]/180*np.pi, np.arccos(galaxies["ba"]),
    galaxies["gama"],
    galaxies["ex"], galaxies["ey"], galaxies["ez"]
))

# Random inclination angle
cos["random"] = np.random.uniform(0, 1, len(galaxies))
phi["random"] = np.random.uniform(0, np.pi, len(galaxies))
dum["random"] = np.concatenate(get_dum(
    galaxies["ra"], galaxies["dec"],
    phi["random"], np.arccos(cos["random"]),
    galaxies["gama"],
    galaxies["ex"], galaxies["ey"], galaxies["ez"]
))

#%% Other methods
for method in ("global", "classifier"):
    galaxies = pd.read_csv("data/intermediate/filament_galaxies_%s.csv" % method)

    cos[method] = np.cos(galaxies["t"])
    phi[method] = galaxies["p"]
    dum[method] = np.concatenate(get_dum(
        galaxies["ra"], galaxies["dec"],
        galaxies["p"], galaxies["t"],
        galaxies["gama"],
        galaxies["ex"], galaxies["ey"], galaxies["ez"]
    ))

#%%
for method in ("random", "global", "classifier"):
    kde_cos = sm.nonparametric.KDEUnivariate(
        np.concatenate((-1*cos[method], cos[method], 2 - cos[method]))
    )
    kde_cos.fit(bw=0.03)
    
    kde_phi = sm.nonparametric.KDEUnivariate(
        np.concatenate((-1*phi[method], phi[method], 2*np.pi - phi[method]))
    )
    kde_phi.fit(bw=0.03)
    
    kde_dum = sm.nonparametric.KDEUnivariate(
        np.concatenate((-1*dum[method], dum[method], 2 - dum[method]))
    )
    kde_dum.fit(bw=0.03)

    plt.figure(1)
    plt.plot(kde_cos.support, kde_cos.density*3, label=method)
    #plt.hist(dum[method], 100, (0, 1), True, histtype="step", label=method)
    
    plt.figure(2)
    plt.plot(kde_phi.support, kde_phi.density*3, label=method)
    
    plt.figure(3)
    plt.plot(kde_dum.support, kde_dum.density*3, label=method)

plt.figure(1)
plt.xlim((0, 1))
plt.gca().legend()

plt.figure(2)
plt.xlim((0, np.pi))
plt.gca().legend()

plt.figure(3)
plt.xlim((0, 1))
plt.ylim((0.85, 1.15))
plt.gca().legend()

#plt.savefig("plots/dum.png")
