#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
from scipy.optimize import fsolve, differential_evolution, brute
from helpers import get_truncnorm_sample, plot_truncnorm_pdf
from time import time
from multiprocessing import Pool
from os import cpu_count
import statsmodels.api as sm
from inclination import cos_t

#%%
def get_ba(p, x, z, cos_t):
    x2 = x**2
    z2 = z**2

    cos2_t = cos_t**2
    sin2_t = 1 - cos2_t

    cos2_p = np.cos(p)**2
    sin2_p = np.sin(p)**2
    sin_2p = np.sin(2*p)

    A = cos2_t/x2 * (sin2_p + cos2_p/z2) + sin2_t/z2
    B = (1 - 1/z2) * 1/x2 * cos_t * sin_2p
    C = (sin2_p/z2 + cos2_p)/x2
    D = np.sqrt((A - C)**2 + B**2)

    return np.sqrt((A + C - D) / (A + C + D))

N = 1000
p = np.random.uniform(0, np.pi*2, N)
ba = np.random.uniform(0, 1, N)

for ba in np.random.uniform(0, 1, N):
    p = np.random.uniform(0, np.pi*2)
    x = np.random.uniform(0, ba)
    z = np.random.uniform(ba, 1)
    cos = cos_t(p, x, z, ba)

    if np.isnan(cos):
        print("nan", p < x, p > z)
    elif (ba - get_ba(p, x, z, cos)) > 0.001:
        print("wrong", p, x, z, ba)
