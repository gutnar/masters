#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin, tan, lambdify, simplify, trigsimp, Abs
from helpers import get_truncnorm_sample
from scipy.optimize import differential_evolution
from time import time
from multiprocessing import Pool
from os import cpu_count

t = Symbol("t", positive=True)
p = Symbol("p", positive=True)
x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)

A = t**2 / x**2 * (sin(p)**2 + cos(p)**2 / z**2) + (1 - t**2) / z**2
B = (1 - 1/z**2) * 1/x**2 * t * sin(2*p)
C = (sin(p)**2/z**2 + cos(p)**2) / x**2
D = sqrt((A - C)**2 + B**2)
Q = (1 - q**2) / (1 + q**2)
E = q**2 * (A + C + D) - (A + C - D)

#%%
cos_t = solve(Eq(Q**2 * (A+C)**2, (A - C)**2 + B**2), t)

#%%
from sympy.abc import psi

solve((
    Eq(Q**2 * (A+C)**2, (A - C)**2 + B**2),
    Eq(tan(2*psi)*(A - C), B)
), (t, p))

#%%
trigsimp(cos_t[1])

#%%
p_sample = np.random.uniform(0, 2*np.pi, 1)[0]

for i in range(4):
    print(cos_t[i].subs({
        p: p_sample,
        x: 0.19,
        z: 1,
        q: 0.1973
    }))

E.subs({
    t: 0.0541588193698194,
    p: p_sample,
    x: 0.19,
    z: 1,
    q: 0.1973
})

#%%
#f = lambdify([])
galaxies = pd.read_csv("data_inclinations.txt")

#%%
t_samples = []

for i in range(50):
    galaxy = galaxies.iloc[[i]]
    q_sample = float(galaxy["ba"])
    x_sample = min(q_sample, float(galaxy["x_mu"]))
    z_sample = max(q_sample, float(galaxy["z_mu"]))
    p_sample = np.random.uniform(0, 2*np.pi, 1)[0]

    t_samples.append(cos_t[1].subs({
        p: p_sample,
        x: x_sample,
        z: z_sample,
        q: q_sample
    }).evalf())

plt.hist(t_samples, 10, (-2, 2))
min(t_samples), max(t_samples)

#%%
expression = sqrt(q**2*x**2*z**2*sin(p)**2/2 - q**2*x**2*z**2/2 - q**2*x**2*sin(p)**2/2 + q**2*z**2/2 + x**4 - x**2*z**2*sin(p)**2 + x**2*sin(p)**2 - x**2 + z**4*sin(p)**4 - z**4*sin(p)**2 - 2*z**2*sin(p)**4 + 2*z**2*sin(p)**2 - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/2 + sin(p)**4 - sin(p)**2 + x**2*z**2*sin(p)**2/(2*q**2) - x**2*z**2/(2*q**2) - x**2*sin(p)**2/(2*q**2) + z**2/(2*q**2) - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/(2*q**2))/Abs(x**2 - z**2*sin(p)**2 + sin(p)**2 - 1)

f = lambdify([p, x, z, q], expression, "numpy")

t_samples = f(
    np.random.uniform(0, 2*np.pi, 50),
    galaxies["x_mu"],
    galaxies["z_mu"],
    galaxies["ba"]
)

plt.hist(t_samples, 10, (0, 1))
min(t_samples), max(t_samples)

#%%
from sympy import factor

factor(expression)
