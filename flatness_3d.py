#%%
from common import pd, np, plt, galaxies_test
from helpers import get_best_estimator


class Flatness3dEstimator:
    mean = 0.5
    deviation = 0

    def __init__(self, x):
        self.mean = x[0]
        self.deviation = x[1]

    def fit(self, target):
        return np.sum(np.square((target - self.bahist())))
    
    def bahist(self, size=50000):
        z = np.random.normal(0.9, 0.05, size)
        z2 = z**2
        x = np.random.normal(self.mean, self.deviation, size)
        x = np.maximum(0.01, x)
        x = np.minimum(z, x)
        x2 = x**2

        cos_t = np.random.uniform(0, 1, size)
        cos2_t = cos_t**2
        sin2_t = 1 - cos2_t

        cos_p = np.random.uniform(0, 1, size)
        cos2_p = cos_p**2
        sin2_p = 1 - cos2_p
        sin_2p = 2*np.sqrt(sin2_p)*np.sqrt(cos2_p)

        A = cos2_t/x2 * (sin2_p + cos2_p/z2) + sin2_t/z2
        B = (1 - 1/z2) * 1/x2 * cos_t * sin_2p
        C = (sin2_p/z2 + cos2_p)/x2
        D = np.sqrt((A - C)**2 + B**2)

        ba = np.sqrt((A + C - D) / (A + C + D))
        baslot = np.ceil(ba * 100) - 1
        bahist = np.histogram(baslot, 100, (0, 100), density=True)[0]
        
        return bahist


def get_flatness_3d(target):
    return get_best_estimator(Flatness3dEstimator, ((0, 1), (0, 1)), target, (0.01, 0.01))


def compare_flatness_hist(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        hist = np.histogram(galaxies[quantiles == i]["baslot"].values, 100, (0, 100), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        flatness = get_flatness_3d(hist)
        
        plt.plot(hist, 'o', color=color, label="("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"]")
        plt.plot(flatness.bahist(100000), color=color, label=("f ~ N(%.2f, %.2f)" % (flatness.mean, flatness.deviation)))
        
        plt.title(parameter)
        plt.gca().legend()
        #plt.savefig("plots/flatness_3d_" + parameter + "_hist.png")

#%%
if __name__ == '__main__':
    compare_flatness_hist(galaxies_test, "rmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_flatness_hist(galaxies_test, "rabsmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_flatness_hist(galaxies_test, "redshift", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_flatness_hist(galaxies_test, "rad", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_flatness_hist(galaxies_test, "sern", (0, 1/2, 1))
