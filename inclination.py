#%%
from common import pd, np, plt, galaxies_test
from helpers import get_best_estimator, get_truncnorm_sample


class InclinationEstimator:
    x_mean = 0.5
    x_dev = 0
    z_mean = 0.9
    z_dev = 0.05

    def __init__(self, x):
        self.x_mean = x[0]
        self.x_dev = x[1]
        self.z_mean = x[2]
        self.z_dev = x[3]

    def fit(self, target):
        return np.sum(np.square((target - self.bahist())))
    
    def sample_x(self, size):
        return get_truncnorm_sample(self.x_mean, self.x_dev, 0, 1, size)
    
    def sample_z(self, size=1):
        return get_truncnorm_sample(self.z_mean, self.z_dev, 0, 1, size)
    
    def bahist(self, size=10000):
        x = self.sample_x(size)
        z = self.sample_z(size)

        x = np.minimum(x, z)
        #z = np.maximum(x, z)

        x2 = x**2
        z2 = z**2

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


def get_inclination(target):
    return get_best_estimator(InclinationEstimator, (
        (0.01, 0.5), (0.01, 0.5), (0.8, 1), (0.01, 0.1)
    ), target, (0.01, 0.01, 0.01, 0.01))


def compare_inclination_hist(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        hist = np.histogram(galaxies[quantiles == i]["baslot"].values, 100, (0, 100), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        inclination = get_inclination(hist)
        
        plt.plot(hist, 'o', color=color, label="("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"]")
        plt.plot(inclination.bahist(100000), color=color,
            label=("f ~ N(%.2f, %.2f), z ~ N(%.2f, %.2f), e = %.2E" %
            (inclination.x_mean, inclination.x_dev, inclination.z_mean, inclination.z_dev, inclination.fit(hist))))
        
        plt.title(parameter)
        plt.gca().legend()
        #plt.savefig("plots/inclination_" + parameter + "_hist.png")

#%%
if __name__ == '__main__':
    compare_inclination_hist(galaxies_test, "rmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_inclination_hist(galaxies_test, "rabsmag", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_inclination_hist(galaxies_test, "redshift", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_inclination_hist(galaxies_test, "rad", (0, 1/2, 1))

#%%
if __name__ == '__main__':
    compare_inclination_hist(galaxies_test, "sern", (0, 1/2, 1))
