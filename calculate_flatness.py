#%%
from common import pd, np, plt, galaxies, galaxies_train, parameters
from classifier import clf
from flatness_3d import get_flatness_3d, compare_flatness_hist

#%%
hist = clf.predict_proba(galaxies.iloc[[1]][parameters].values)
flatness = get_flatness_3d(hist)

plt.plot(list(hist)[0], 'o')
plt.plot(flatness.bahist(100000), label=("f ~ N(%.2f, %.2f)" % (flatness.mean, flatness.deviation)))
plt.gca().legend()

#%%
sample = galaxies[:10]
sample.describe()

for i in range(len(sample)):
    galaxy = sample.iloc[[i]]
    hist = clf.predict_proba(galaxy[parameters].values)
    flatness = get_flatness_3d(hist)

    print(flatness.mean, flatness.deviation)
