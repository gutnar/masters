#%%
import numpy as np
import pandas as pd

from lib import BayesianApproximation, PDF
from lib.plotting import *

#%%
galaxies = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
#galaxies = pd.read_csv("data/intermediate/elliptic_galaxies.csv")

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
#ba.Z_MIN = 0.99
#ba.run([(1000, "scott")]*100)
ba.run([(150000, 0.05)]*25)
print(time() - start)

plot_ba_results(ba)

#%%
plot_xz_kde(ba)

#%%
plot_qt_kde(ba)

#%%
plot_qp_kde(ba)

#%%
plt.title(r"$\theta$")

for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    t_pdf = ba.get_t_pdf(q)
    plt.plot(t_pdf.x, t_pdf.y, label=r"$q = %.2f$" % q)

plt.legend()

#%%
plt.title(r"$\phi$")

for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    p_pdf = ba.get_p_pdf(q)
    plt.plot(p_pdf.x, p_pdf.y, label=r"$q = %.2f$" % q)

plt.legend()

#%%
for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    x_pdf = ba.get_x_pdf(q)
    plt.plot(x_pdf.x, x_pdf.y, label=r"$q = %.2f$" % q)
    plt.legend()

#%%
q = 0.1
x_pdf = ba.get_x_pdf(q)
x = x_pdf.sample(10000)
plt.hist(np.arccos(
    np.sqrt(np.maximum(0, (q**2 - x**2)/(1 - x**2)))
), 100, (-np.pi/2, np.pi/2))

#%%
plt.hist(p_pdf.sample(10000), 100, (-np.pi/2, np.pi/2), True)
plt.plot(p_pdf.x, p_pdf.y * 8)

#%%
plot_tp_kde(ba, 0.9)
