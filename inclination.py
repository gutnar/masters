#%%
from common import pd, np, plt, galaxies_test
from helpers import get_best_estimator, get_truncnorm_sample, get_ba
from scipy import stats
from time import time
from scipy.optimize import differential_evolution, minimize
import statsmodels.api as sm


class InclinationEstimator:
    x_mean = 0.5
    x_dev = 0.1
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
    
    def sample_ba(self, size):
        x = self.sample_x(size)
        z = self.sample_z(size)

        #x = np.minimum(x, z)
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

        return np.sqrt((A + C - D) / (A + C + D))
    
    def bahist(self, size=1000):
        ba = self.sample_ba(size)
        #baslot = np.ceil(ba * 100) - 1
        #bahist = np.histogram(baslot, 100, (0, 100), density=True)[0]
        
        kde = stats.gaussian_kde(ba)
        bahist = kde(np.linspace(0, 1, 100))
        
        return bahist


def get_inclination(target):
    return get_best_estimator(InclinationEstimator, (
        (0.01, 0.51), (0.01, 0.51), (0.8, 1), (0.01, 0.11)
    ), target, (0.01, 0.01, 0.01, 0.01))
    #return get_best_estimator(InclinationEstimator, (
    #    (0.01, 0.5), (0.7, 1)
    #), target, (0.01, 0.01))


def test_parameters(parameters, target, N):
    x = get_truncnorm_sample(parameters[0], parameters[1], 0, 1, N)
    #z = get_truncnorm_sample(parameters[2], parameters[3], 0, 1, N)
    z = get_truncnorm_sample(0.9, 0.006, 0, 1, N)
    ba = get_ba(x, z)
    #kde = stats.gaussian_kde(ba)
    #ba_hist = np.histogram(ba, 100, (0, 1), density=True)[0]
    #kde(np.linspace(0, 1, 100))

    kde = sm.nonparametric.KDEUnivariate(ba)
    kde.fit()
    ba_hist = kde.evaluate(np.linspace(0, 1, 100))

    return np.sum(np.square((target - ba_hist)))


def compare_inclination_hist(galaxies, parameter, cuts):
    quantiles = pd.qcut(galaxies[parameter], cuts, labels=False)

    for i in range(len(cuts) - 1):
        hist = np.histogram(galaxies[quantiles == i]["ba"].values, 100, (0, 1), density=True)[0]
        color = next(plt.gca()._get_lines.prop_cycler)['color']

        inclination = get_inclination(hist)
        #result = (inclination.x_mean, inclination.x_dev, inclination.z_mean, inclination.z_dev)
        #result = differential_evolution(test_parameters, ((0.01, 0.5), (0, 0.25), (0.7, 1), (0, 0.15)), args=(hist, 100), maxiter=25)
        result = differential_evolution(test_parameters, ((0.1, 0.4), (0, 0.25)), args=(hist, 1000), maxiter=50)
        #result = minimize(test_parameters, (0.25, 0.1, 0.85, 0.05), (hist, 100), "L-BFGS-B", bounds=(
        #    (0, 0.5), (0, 0.25), (0.7, 1), (0, 0.15)
        #), options={
        #    "ftol": 0.01
        #}).x
        print(result)
        result = result.x

        x = get_truncnorm_sample(result[0], result[1], 0, 1, 10000)
        #z = get_truncnorm_sample(result[2], result[3], 0, 1, 10000)
        z = get_truncnorm_sample(0.9, 0.006, 0, 1, 10000)
        ba = get_ba(x, z)
        kde = stats.gaussian_kde(ba)
        
        ba_slot = np.linspace(0, 1, 100)
        plt.plot(ba_slot, hist, 'o', color=color, label="("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"]")
        plt.plot(ba_slot, kde(ba_slot), color=color)
        
        '''
        inclination = get_inclination(hist)
        ba_slot = np.linspace(0, 1, 100)

        plt.plot(ba_slot, hist, 'o', color=color, label="("+str(round(cuts[i], 2))+", "+str(round(cuts[i+1], 2))+"]")
        plt.plot(ba_slot, inclination.bahist(), color=color,
            label=("f ~ N(%.2f, %.2f), z ~ N(%.2f, %.2f), e = %.2E" %
            (inclination.x_mean, inclination.x_dev, inclination.z_mean, inclination.z_dev, inclination.fit(hist))))
        
        #ba = inclination.sample_ba(10000)
        #kde = stats.gaussian_kde(ba)
        #plt.plot(ba_slot, kde(ba_slot))
        
        plt.title(parameter)
        plt.gca().legend()
        #plt.savefig("plots/inclination_" + parameter + "_hist.png")
        '''

#%%
x_mu = (0, 0.5)
x_sigma = (0, 0.25)
z_mu = (0.7, 1)
z_sigma = (0, 0.15)

kde = sm.nonparametric.KDEUnivariate(galaxies_test["ba"].values)
kde.fit()

plt.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)

plt.plot(np.linspace(0, 1, 100), kde.evaluate(np.linspace(0, 1, 100)), lw=3, label='KDE from samples', zorder=10)

#%%
if __name__ == '__main__':
    start = time()
    compare_inclination_hist(galaxies_test, "rmag", (0, 1))
    print(time() - start)

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
