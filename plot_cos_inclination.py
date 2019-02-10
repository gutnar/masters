#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
from scipy.optimize import fsolve, differential_evolution
from helpers import get_truncnorm_sample
from time import time
from multiprocessing import Pool

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
    #print(p_value, x_value, z_value, q_value, cos_t[0])
    #print(E.evalf(subs={ p: p_value, x: x_value, z: z_value, q: q_value, t: cos_t[0] }))
    return E2.subs({
        #p: p_value,
        x: x_value,
        z: z_value,
        q: q_value,
        t: cos[0],
        p: cos[1]
    })

def process_galaxies(data):
    pass

#%%
galaxies = pd.read_csv("data_inclinations.1000.txt")
galaxies.describe()

plt.hist(galaxies[:100]["baslot"], 100, (0, 100))

#%%
start = time()
sample_size = 1
cos_samples = []

for i in range(100):
    galaxy = galaxies.iloc[[i]]
    ba = float(galaxy["ba"])

    sample_x = get_truncnorm_sample(galaxy["x_mu"], galaxy["x_sigma"], 0, 1, sample_size)
    sample_z = get_truncnorm_sample(galaxy["z_mu"], galaxy["z_sigma"], 0, 1, sample_size)
    #sample_p = np.random.uniform(0, 1, sample_size)

    #sample_x = np.minimum(sample_x, sample_z)
    #x = np.minimum(x, ba*z)
    #cos_x = np.sqrt((ba**2 * z**2 - x**2) / (1 - x**2))

    '''
    cos_t_samples += [
        fsolve(lambda test_t: E.subs({
            p: sample_p[i],
            x: sample_x[i],
            z: sample_z[i],
            q: ba,
            t: test_t
        }), 0.5) for i in range(sample_size)
    ]
    '''

    cos_samples += [
        differential_evolution(
            test_cos_t, ((0, 1), (0, 1)),
            (sample_x[i], sample_z[i], ba),
            'best1bin', 1
        ).x
        for i in range(sample_size)
    ]

cos_t_samples = [cos[0] for cos in cos_samples]
cos_t_pdf = np.histogram(np.array(cos_t_samples), 100, (0, 1))[0]
#cos_t_pdf = cos_t_pdf[1:]
#rand_cos_t_pdf = np.histogram(np.array(rand_cos_t_samples), 100, (0, 1))[0]

#x_pdf /= len(galaxies) * 100
#x_rand_pdf /= len(galaxies) * 100

#err = np.sum((x_pdf - np.mean(x_pdf))**2)
#err_rand = np.sum((x_rand_pdf - np.mean(x_rand_pdf))**2)

plt.plot(cos_t_pdf)
#plt.plot(rand_cos_t_pdf)
#plt.plot(x_rand_pdf, label="N(0.14, 0.05), e = %.2E" % err_rand)
#plt.legend()

print(time() - start)

#%%
g = galaxies.iloc[[0]]

start = time()
#print(fsolve(test_cos_t, [0.5, 0.5], args=(0.32, 0.89, 0.40)))
#print(shgo(test_cos_t, ((0, 1), (0, 1)), (0.32, 0.89, 0.40)))
#print(root(test_cos_t, (0.5, 0.5), (0.32, 0.89, 0.40)))
#print(newton(test_cos_t, (0.5, 0.5), None, (0.32, 0.89, 0.40)))
print(differential_evolution(
    test_cos_t, ((0, 1), (0, 1)),
    #(0.32, 0.89, 0.40),
    (g["x_mu"], g["z_mu"], g["ba"]),
    'best1bin', 1
).x)
print(time() - start)

# (0.23, 0.72)