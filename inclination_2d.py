#%%
from common import pd, np, plt, galaxies_test
from helpers import get_best_estimator


class Inclination2dEstimator:
    mean = 0.5
    dev = 0
    
    def __init__(self, x):
        self.mean = x[0]
        self.dev = x[1]

    def fit(self, target):
        return np.sum(np.square((target - self.bahist())))
    
    def bahist(self, size=10000):
        f = np.random.normal(self.mean, self.dev, size)
        f = np.maximum(0.01, f)
        cos = np.random.uniform(0, 1, size)

        ba = np.sqrt(cos**2 * (1 - f**2) + f**2)
        baslot = np.ceil(ba * 100) - 1
        bahist = np.histogram(baslot, 100, (0, 100), density=True)[0]
        
        return bahist


def get_inclination_2d(target):
    return get_best_estimator(Inclination2dEstimator, ((0, 1), (0, 1)), target, (0.01, 0.01))


def compare_inclination_hist(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        hist = np.histogram(galaxies[quantiles == i]["baslot"].values, 100, (0, 100), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        inclination = get_inclination_2d(hist)
        
        plt.plot(hist, 'o', color=color, label="("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"]")
        plt.plot(inclination.bahist(100000), color=color, label=("f ~ N(%.2f, %.2f)" % (inclination.mean, inclination.dev)))
        
        plt.title(parameter)
        plt.gca().legend()

#%%
compare_inclination_hist(galaxies_test, "rmag", (0, 1/2, 1))

#%%
compare_inclination_hist(galaxies_test, "rabsmag", (0, 1/2, 1))

#%%
compare_inclination_hist(galaxies_test, "redshift", (0, 1/2, 1))

#%%
compare_inclination_hist(galaxies_test, "rad", (0, 1/2, 1))

#%%
compare_inclination_hist(galaxies_test, "sern", (0, 1/2, 1))