#%%
import pandas as pd
import matplotlib.pyplot as plt

from lib import PDF, BayesianApproximation2d
from lib.plotting import *
from src.tex_plot import savefig

#%%
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
#galaxies = galaxies[galaxies["sern"] < 2]

q_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    galaxies["ba"].values
)

plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
plt.plot(q_pdf.x, q_pdf.y)

#%%
ba = BayesianApproximation2d(q_pdf, 150000)
ba.run([(1000, "scott")]*100 + [(150000, "scott")]*25)

plot_ba_2d_results(ba)

#%%
plot_xz_kde(ba)

plt.tick_params(direction="in")
#plt.savefig("plots/xi_zeta_intial_kde.pdf", dpi=1000, bbox_inches='tight')#, pad_inches=0)

#%%
plot_q_theta_kde(ba)

#%%
plot_q_phi_kde(ba)

#%%
plot_pos_inc_kde(ba, 0.9, np.pi/4)

#%%
q, xi, zeta, theta, phi = ba.sample(10000)

#%%
plot_kde(ba.xz_kde, XZ_GRID)

#%%
import numpy as np
mu, sigma = 60, 10 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

ga = pd.DataFrame({'r':s})

cm = ga['r'].cumsum()
plt.hist(ga['r'], bins=100)
plt.plot(cm.values,cm.index)
#%%
ga.loc[ga.idxmax()]

#%%
