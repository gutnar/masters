#%%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from lib import PDF, Classifier, BayesianApproximation2d
from lib.plotting import *
from src.tex_plot import savefig


def get_truncnorm_sample(mu, sigma, a, b, N):
    return stats.truncnorm.rvs((a - mu)/sigma, (b - mu)/sigma, loc=mu, scale=sigma, size=N)

#%%
galaxies = pd.read_csv("data/intermediate/galaxies.csv")
train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")
test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")
#galaxies = galaxies[galaxies["sern"] < 2]

q_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    galaxies["ba"].values
)

plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
plt.plot(q_pdf.x, q_pdf.y)

#%% Manual fit
classifier = Classifier()
classifier.fit(train_galaxies)

#%%
ba = BayesianApproximation2d(q_pdf)
ba.run()

#%%
plot_ba_2d_results(ba)

#%%
plot_xz_kde(ba)
#plt.tick_params(direction="in")
#plt.savefig("plots/xi_zeta_intial_kde.pdf", dpi=1000, bbox_inches='tight')#, pad_inches=0)

#%%
pos, inc = ba.sample_pos_inc(0.1, 0.5)

#plt.hist(np.abs(pos), 100, label="pos", histtype="step")
plt.hist(np.abs(inc), 100, (0, 1), label="inc", histtype="step")

plt.legend()

#%%
q = 0.05
bw = 0.005

q_sample, xi, zeta, theta, phi = ba.sample(1000000)
sample = (q_sample > (q - bw)) & (q_sample < (q + bw))

plt.hist(xi[sample], 100, label="xi", histtype="step")
plt.hist(zeta[sample], 100, label="zeta", histtype="step")
plt.hist(np.abs(np.cos(theta[sample])), 100, label="theta", histtype="step")

plt.legend()

#%%
from lib.binney import get_q

plt.hist(get_q(xi2, zeta2, theta2, phi2), 100, histtype="step", label="2")
plt.hist(get_q(xi[sample], zeta[sample], theta[sample], phi[sample]), 100, histtype="step", label="1")
plt.legend()

#%%
plt.hist(xi2[(np.abs(np.cos(theta2)) > 0.9) & (xi2 > 0.04)], 100, histtype="step", label="xi")
plt.legend()

#%%
weird = (np.abs(np.cos(theta2)) > 0.95) & (xi2 > 0.0445)

xi2[weird][0], zeta2[weird][0], theta2[weird][0], phi2[weird][0]

#%%


#%%
plot_q_theta_kde(ba)

#%%
plot_q_phi_kde(ba)

#%%
plot_pos_inc_kde(ba, 1, np.pi/4)

#%%
q, xi, zeta, theta, phi = ba.sample(10000)

#%%
plot_kde(ba.xz_kde, XZ_GRID)
