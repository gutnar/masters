#%%
import pandas as pd
import matplotlib.pyplot as plt

from lib import PDF, BayesianApproximation2d
from lib.plotting import *

#%%
import matplotlib as mpl

mpl.style.use("default")

plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 11,
    'font.family': 'lmodern',
    'text.latex.unicode': True,
})

#%%
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
galaxies = galaxies[galaxies["g_class"] == 1]

q_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    galaxies["ba"].values
)

plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
plt.plot(q_pdf.x, q_pdf.y)

#%%
ba = BayesianApproximation2d(q_pdf, 150000)
ba.run([(150000, "scott")]*10)

plot_ba_2d_results(ba)

#%%
plot_xz_kde(ba, False)

plt.tick_params(direction="in")
plt.savefig("plots/xi_zeta_intial_kde.pdf", dpi=1000, bbox_inches='tight')#, pad_inches=0)

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

