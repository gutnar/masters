#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import get_truncnorm_sample

#%%
galaxies = pd.read_csv("data_inclinations.txt")
galaxies.describe()

#%%
x_pdf = np.zeros(100)
x_rand_pdf = np.zeros(100)

for i in range(len(galaxies)):
    galaxy = galaxies.iloc[[i]]
    ba = float(galaxy["ba"])

    x = get_truncnorm_sample(galaxy["x_mu"], galaxy["x_sigma"], 0, 1, 100)
    z = get_truncnorm_sample(galaxy["z_mu"], galaxy["z_sigma"], 0, 1, 100)
    
    x = np.minimum(x, ba*z)
    cos_x = np.sqrt((ba**2 * z**2 - x**2) / (1 - x**2))
    x_pdf += np.histogram(cos_x, 100, (0, 1))[0]
    
    # Random
    x = get_truncnorm_sample(0.14, 0.06, 0, 1, 100)
    z = 1
    x = np.minimum(x, ba*z)
    cos_x = np.sqrt((ba**2 * z**2 - x**2) / (1 - x**2))
    x_rand_pdf += np.histogram(cos_x, 100, (0, 1))[0]

x_pdf /= len(galaxies) * 100
x_rand_pdf /= len(galaxies) * 100

err = np.sum((x_pdf - np.mean(x_pdf))**2)
err_rand = np.sum((x_rand_pdf - np.mean(x_rand_pdf))**2)

plt.plot(x_pdf, label="Estimate, e = %.2E" % err)
plt.plot(x_rand_pdf, label="N(0.14, 0.05), e = %.2E" % err_rand)
plt.legend()
#plt.plot(z_pdf)


#A = t^2 / x^2 / z^2 + (1 - t^2) / z^2
#C = 1/x^2

#q^2 = (A + C - |A - C|) / (A + C + |A - C|)
#A = C * q^2
#A = C/q^2

#cos^2(t) = (q^2*z^2 - x^2) / (1 - x^2)
#cos^2(t) = (q^2*x^2 - z^2) / (q^2*x^2 - q^2)

# A = t^2 / x^2 * (1 - p^2 + p^2 / z^2) + (1 - t^2) / z^2
# B = (1 - 1/z^2) * 1/x^2 * t * 2 * sqrt(1 - p^2) * p
# C = ((1 - p^2)/z^2 + p^2) / x^2
# D = sqrt((A - C)^2 + B^2)
# q^2 = (A + C - D) / (A + C + D)

#%%
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin

t = Symbol("t", positive=True)
p = Symbol("p", positive=True)
x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)

A = t**2 / x**2 * (1 - p**2 + p**2 / z**2) + (1 - t**2) / z**2
B = (1 - 1/z**2) * 1/x**2 * t * 2 * sqrt(1 - p**2) * p
C = ((1 - p**2)/z**2 + p**2) / x**2
D = sqrt((A - C)**2 + B**2)

#solve(Eq(q**2, (A + C - D)), t)
solve(Eq(q**2, (A + C - D) / (A + C + D)), t)
