#%%
import pandas as pd
import matplotlib.pyplot as plt

from lib import PDF, BayesianApproximation2d
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
ba = BayesianApproximation2d(q_pdf)
ba.run()

plot_ba_2d_results(ba)

#%%
plot_xz_kde(ba)

#%%
plot_q_theta_kde(ba)

#%%
plot_q_phi_kde(ba)

#%%
plot_pos_inc_kde(ba, 0.8, np.pi/4)
