#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from analytical import get_dum

#%%
galaxies = pd.read_csv("data/intermediate/filament_galaxies_sample.csv")

dum = np.array([])
dum_simple = np.array([])
dum_random = np.array([])
index = 0

for i, galaxy in galaxies.iterrows():
    index += 1

    dum = np.concatenate((
        dum, get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["p"] * 180/np.pi, (np.cos(galaxy["t"]), ),
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ), axis=None)

    # Simple flatness estimation
    dum_simple = np.concatenate((
        dum_simple, get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["pos"], (galaxy["ba"], ),
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ))

    # Random inclination angle
    dum_random = np.concatenate((
        dum_random, get_dum(
            galaxy["ra"], galaxy["dec"], np.random.uniform(-90, 90, 10), np.random.uniform(0, 1, 10),
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ))

#%%
plt.xlim((0, 1))
plt.ylim((0.8, 1.2))
#plt.hist(dum_random, 100, (0, 1), True, histtype="step")
#plt.hist(dum_simple, 100, (0, 1), True, histtype="step")
plt.hist(dum, 50, (0, 1), True, histtype="step")

#%%
kde = sm.nonparametric.KDEUnivariate(dum)
kde_simple = sm.nonparametric.KDEUnivariate(dum_simple)
kde_random = sm.nonparametric.KDEUnivariate(dum_random)

kde.fit(bw=0.01, cut=0)
kde_simple.fit(bw=0.01, cut=0)
kde_random.fit(bw=0.01, cut=0)

plt.plot(kde_random.support, kde_random.density, label="random")
plt.plot(kde_simple.support, kde_simple.density, label="b/a")
plt.plot(kde.support, kde.density, label="bayes")
plt.legend()

#plt.savefig("plots/dum.png")
