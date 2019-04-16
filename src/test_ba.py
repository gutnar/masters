#%%
import numpy as np
import pandas as pd

from lib import BayesianApproximation, PDF, get_dum
from lib.plotting import *

#%%
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")

q_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    galaxies["ba"].values
)

plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
plt.plot(q_pdf.x, q_pdf.y)

#%%
from time import time

start = time()
ba = BayesianApproximation(q_pdf)
#ba.run([(1000, "scott")]*100)
ba.run([(150000, 0.05)]*25)
print(time() - start)

#%%
plot_ba_results(ba)

#%%
plot_xz_kde(ba)

#%%
plot_qt_kde(ba)

#%%
plot_qp_kde(ba)

#%%
t_pdf = ba.get_t_pdf(0.8)
plt.plot(t_pdf.x, t_pdf.y, label=r"$\theta$")
plt.legend()

#%%
p_pdf = ba.get_p_pdf(0.8)
plt.plot(p_pdf.x, p_pdf.y, label=r"$\phi$")
plt.legend()

#%%
N = 10000
gals = pd.read_csv("data/intermediate/filament_galaxies.csv")
galaxy = gals.loc[np.repeat(1, N)]

dum = np.concatenate(get_dum(
    galaxy["ra"], galaxy["dec"],
    p_pdf.sample(N), t_pdf.sample(N),
    galaxy["gama"],
    galaxy["ex"], galaxy["ey"], galaxy["ez"]
))

dum_hist = np.zeros(100)
dum_hist += np.histogram(dum, 100, (0, 1))[0]

plt.plot(dum_hist)

#%%
for i, galaxy in gals.iterrows():
    print(galaxy.index)
    break
