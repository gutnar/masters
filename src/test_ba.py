#%%
import numpy as np
import pandas as pd

from lib import BayesianApproximation, PDF
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
t_pdf = ba.get_t_pdf(0.1)
plt.plot(t_pdf.x, t_pdf.y, label=r"$\theta$")
plt.legend()

#%%
p_pdf = ba.get_p_pdf(0.1)
plt.plot(p_pdf.x, p_pdf.y, label=r"$\phi$")
plt.legend()

#%%
plt.hist(p_pdf.sample(10000), 100, (-np.pi/2, np.pi/2), True)
plt.plot(p_pdf.x, p_pdf.y * 8)
