#%%
from common import pd, np, plt, galaxies, galaxies_train, parameters
from classifier import clf
from inclination import get_inclination
from time import time
import matplotlib.mlab as mlab

#%%
sample = galaxies[:10]
sample.describe()

start = time()

x_mean = 0
x_var = 0

for i in range(1):
    galaxy = sample.iloc[[i]]
    hist = clf.predict_proba(galaxy[parameters].values)
    inclination = get_inclination(hist)

    x_mean += inclination.x_mean
    x_var += inclination.x_dev**2

    print(inclination.x_mean, inclination.x_dev, inclination.z_mean, inclination.z_dev)

x_dev = np.sqrt(x_var)
x = np.linspace(0, 1, 100)
plt.plot(x, mlab.normpdf(x, x_mean, x_dev))

end = time()

end - start
