#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import get_truncnorm_sample, sample_x_slot, sample_z_slot, sample_ba_hist
from sklearn.ensemble import RandomForestClassifier
from time import time

data = pd.read_csv("data_generated.txt")

#%%
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
from scipy.optimize import fsolve, differential_evolution, brute
from helpers import get_truncnorm_sample
from time import time

t = Symbol("t", positive=True)
p = Symbol("p", positive=True)
x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)

A = t**2 / x**2 * (1 - p**2 + p**2 / z**2) + (1 - t**2) / z**2
B = (1 - 1/z**2) * 1/x**2 * t * 2 * sqrt(1 - p**2) * p
C = ((1 - p**2)/z**2 + p**2) / x**2
D = sqrt((A - C)**2 + B**2)
E = (A + C - D) - q**2 * (A + C + D)
E2 = E**2

def test_cos_t(cos, x_value, z_value, q_value):
    return E2.subs({
        x: x_value,
        z: z_value,
        q: q_value,
        t: cos[0],
        p: cos[1]
    })
'''

#%%
err = []
N = 100

for i in range(10000):
    galaxy = data.iloc[[i]]

    x = sample_x_slot(galaxy["x_slot"], N)
    z = sample_z_slot(galaxy["z_slot"], N)
    #ba = sample_ba_hist(galaxy.values[0][2:], N)

    x2 = x**2
    z2 = z**2

    cos_t = np.random.uniform(0, 1, N)
    cos2_t = cos_t**2
    sin2_t = 1 - cos2_t

    cos_p = np.random.uniform(0, 1, N)
    cos2_p = cos_p**2
    sin2_p = 1 - cos2_p
    sin_2p = 2*np.sqrt(sin2_p)*np.sqrt(cos2_p)

    A = cos2_t/x2 * (sin2_p + cos2_p/z2) + sin2_t/z2
    B = (1 - 1/z2) * 1/x2 * cos_t * sin_2p
    C = (sin2_p/z2 + cos2_p)/x2
    D = np.sqrt((A - C)**2 + B**2)

    ba = np.sqrt((A + C - D) / (A + C + D))
    ba_hist = np.histogram(ba, 100, (0, 100))[0] / N

    err += [
        np.sum(np.square(galaxy.values[0][3:] - ba_hist))
    ]

    
    #cos_t += list(np.sqrt((ba_sample**2 * z_sample**2 - x_sample**2) / (1 - x_sample**2)))
    #result = differential_evolution(test_cos_t, ((0, 1), (0, 1)), (x_sample[0], z_sample[0], ba_sample[0]), 'best1bin', 1).x
    #print(i, result)
    #cos_t += [result[0]]

plt.hist(err, 100)
