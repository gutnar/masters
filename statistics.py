#%%
from common import pd, np, plt, galaxies_test
from helpers import get_best_estimator, get_truncnorm_sample, get_ba
from scipy import stats
from time import time
from scipy.optimize import differential_evolution, minimize
import statsmodels.api as sm

def test_parameters(parameters, target, N):
    x = get_truncnorm_sample(parameters[0], parameters[1], 0, 1, N)
    z = get_truncnorm_sample(parameters[2], parameters[3], 0, 1, N)
    ba = get_ba(x, z)

    kde = sm.nonparametric.KDEUnivariate(ba)
    kde.fit()
    ba_hist = kde.evaluate(np.linspace(0, 1, 100))

    return np.sum(np.square((target - ba_hist)))


#%%
target = np.histogram(galaxies_test["ba"].values, 100, (0, 1), density=True)[0]
result = differential_evolution(test_parameters, ((0.1, 0.4), (0, 0.25), (0.8, 1), (0, 0.1)), args=(target, 1000))

print(result)

#fun: 0.2699892181162029
#message: 'Maximum number of iterations has been exceeded.'
#    nfev: 60225
#    nit: 1000
#success: False
#    x: array([0.29141139, 0.23187484, 0.93228061, 0.07318355])

#%%
x = get_truncnorm_sample(result.x[0], result.x[1], 0, 1, 10000)
z = get_truncnorm_sample(result.x[2], result.x[3], 0, 1, 10000)
ba = get_ba(x, z)
kde = stats.gaussian_kde(ba)

ba_slot = np.linspace(0, 1, 100)
plt.plot(ba_slot, target, "o")
plt.plot(ba_slot, kde(ba_slot))
