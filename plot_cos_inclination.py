#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
from scipy.optimize import fsolve, minimize
from helpers import get_truncnorm_sample

t = Symbol("t", positive=True)
p = Symbol("p", positive=True)
x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)

A = t**2 / x**2 * (1 - p**2 + p**2 / z**2) + (1 - t**2) / z**2
B = (1 - 1/z**2) * 1/x**2 * t * 2 * sqrt(1 - p**2) * p
C = ((1 - p**2)/z**2 + p**2) / x**2
D = sqrt((A - C)**2 + B**2)
E = (A + C - D) / (A + C + D) - q**2

#%%
galaxies = pd.read_csv("data_inclinations.1000.txt")
galaxies.describe()

#%%
sample_size = 1
cos_t_samples = []
rand_cos_t_samples = []

for i in range(5):
    galaxy = galaxies.iloc[[i]]
    ba = float(galaxy["ba"])

    sample_x = get_truncnorm_sample(galaxy["x_mu"], galaxy["x_sigma"], 0, ba, 1)[0]
    sample_z = get_truncnorm_sample(galaxy["z_mu"], galaxy["z_sigma"], ba, 1, 1)[0]
    sample_p = np.random.uniform(0, 1, 1)[0]
    
    #sample_x = np.minimum(sample_x, sample_z)
    #x = np.minimum(x, ba*z)
    #cos_x = np.sqrt((ba**2 * z**2 - x**2) / (1 - x**2))

    cos_t = fsolve(lambda test_t: E.subs({ p: sample_p, x: sample_x, z: sample_z, q: ba, t: test_t }), 0.5)
    cos_t_samples.append(cos_t)
    #x_pdf += np.histogram([cos_t], 100, (0, 1))[0]
    
    # Random
    sample_x = get_truncnorm_sample(0.14, 0.06, 0, ba, 1)[0]
    cos_t = (ba**2 - sample_x**2) / (1 - sample_x**2)
    rand_cos_t_samples.append(cos_t)
    #x_rand_pdf += np.histogram(cos_x, 100, (0, 1))[0]

x_pdf = np.histogram(np.array(cos_t_samples), 100, (0, 1))[0]
x_rand_pdf = np.histogram(np.array(rand_cos_t_samples), 100, (0, 1))[0]

#x_pdf /= len(galaxies) * 100
#x_rand_pdf /= len(galaxies) * 100

err = np.sum((x_pdf - np.mean(x_pdf))**2)
err_rand = np.sum((x_rand_pdf - np.mean(x_rand_pdf))**2)

plt.plot(x_pdf, label="Estimate, e = %.2E" % err)
plt.plot(x_rand_pdf, label="N(0.14, 0.05), e = %.2E" % err_rand)
plt.legend()

#%%
def test_cos_t(cos_t, p_value, x_value, z_value, q_value):
    #print(p_value, x_value, z_value, q_value, cos_t[0])
    #print(E.evalf(subs={ p: p_value, x: x_value, z: z_value, q: q_value, t: cos_t[0] }))
    return E.subs({ p: p_value, x: x_value, z: z_value, q: q_value, t: cos_t[0] })

fsolve(test_cos_t, 0.2, args=(0.396, 0.2, 0.9, 0.507))#, bounds=[(0, 1)])
