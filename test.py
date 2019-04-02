#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, sqrt, solve, Eq, cos, sin
from scipy.optimize import fsolve, differential_evolution, brute
from helpers import PDF
from time import time
from multiprocessing import Pool
from os import cpu_count
import statsmodels.api as sm
import scipy.stats as stats

#%%
galaxies = pd.read_csv("data/intermediate/inclinations.txt")

#%%
# Create some dummy data
rvs = np.append(stats.norm.rvs(loc=2,scale=1,size=(1000,1)),
                stats.norm.rvs(loc=0,scale=3,size=(1000,1)),
                axis=1)

# Regular grid to evaluate kde upon
x_flat = np.r_[rvs[:,0].min():rvs[:,0].max():128j]
y_flat = np.r_[rvs[:,1].min():rvs[:,1].max():128j]
x,y = np.meshgrid(x_flat,y_flat)
grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

#%%
start = time()

kde = stats.kde.gaussian_kde(rvs.T)
z = kde(grid_coords.T)
z = z.reshape(128,128)

plt.imshow(z,aspect=x_flat.ptp()/y_flat.ptp())

time() - start


#%%
ba_kde = sm.nonparametric.KDEUnivariate(galaxies["ba"].values)
ba_kde.fit(bw=0.01)
ba_grid = np.linspace(0, 1, 100)
ba_pdf = PDF(ba_kde.evaluate(ba_grid), ba_grid)

plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
ba_pdf.plot()

#%%
samples = ba_pdf.sample(100000)
plt.hist(samples, 100, (0, 1), density=True)
ba_pdf.plot()

#%%
ba_samples = ba_pdf.sample(100)

#%%
def bayesian(ba_pdf):
    pass


#%%
qx = np.column_stack((
    ba_pdf.sample(50000),
    np.random.uniform(0, 1, 50000)
))

x, y = np.meshgrid(
    np.linspace(0, 1, 100),
    np.linspace(0, 1, 100)
)

qx_grid = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

#%%
start = time()
qx_kde = stats.kde.gaussian_kde(qx.T)
qx_pdf = qx_kde(qx_grid.T).reshape(100, 100)
print(time() - start)

plt.imshow(qx_pdf, aspect=1)

#%%
qx2 = qx[qx[:,0] < qx[:,1]]

#%%
start = time()
qx_kde = sm.nonparametric.KDEMultivariate(qx.T, "cc")
qx_pdf = qx_kde.pdf(qx_grid.T).reshape(100, 100)
print(time() - start)

plt.imshow(qx_pdf, aspect=1)

#%%
plt.plot(qx_pdf.reshape(100, 100)[:,15])
