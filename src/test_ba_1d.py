#%%
import pandas as pd
import matplotlib.pyplot as plt

from lib import PDF, BayesianApproximation1d
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
ba = BayesianApproximation1d(q_pdf)
ba.run()#[(150000, 0.05)]*25)

plot_ba_1d_results(ba)

#%%
plot_qi_kde(ba)
