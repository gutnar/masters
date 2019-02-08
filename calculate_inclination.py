#%%
from common import pd, np, plt, galaxies, galaxies_train, parameters
from classifier import clf
from inclination import get_inclination
from time import time

#%%
hist = clf.predict_proba(galaxies.iloc[[1]][parameters].values)
inclination = get_inclination(hist)

plt.plot(list(hist)[0], 'o')
plt.plot(inclination.bahist(100000), label=("f ~ N(%.2f, %.2f)" % (inclination.x_mean, inclination.x_dev)))
plt.gca().legend()

#%%
sample = galaxies[:10]
sample.describe()

start = time()

for i in range(1):
    galaxy = sample.iloc[[i]]
    hist = clf.predict_proba(galaxy[parameters].values)
    inclination = get_inclination(hist)

    print(inclination.x_mean, inclination.x_dev)

end = time()

end - start
