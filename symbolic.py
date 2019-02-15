#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
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
E = q**2 - (A + C - D) / (A + C + D)
E2 = E**2

#%%
def test_cos_t(cos, p_value, x_value, z_value, q_value):
    return E2.subs({
        t: cos[0],
        p: p_value,
        x: x_value,
        z: z_value,
        q: q_value
    })

#start = time()
#print(differential_evolution(test_cos_t, [
#        (0, 1)
#    ], args=(0.88, 0.32, 0.89, 0.5), maxiter=1))
#time() - start



#fsolve(test_cos_t, 0.5, (0.88, 0.32, 0.89, 0.33))