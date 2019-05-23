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
#train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")
#test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")

spiral = galaxies[galaxies["sern"] < 2]
elliptic = galaxies[galaxies["sern"] > 2]

spiral_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    spiral["ba"]
)

elliptic_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    elliptic["ba"]
)

#plt.hist(galaxies["ba"].values, 100, (0, 1), density=True)
plt.plot(spiral_pdf.x, spiral_pdf.y)
plt.plot(elliptic_pdf.x, elliptic_pdf.y)

#%% Manual fit
#classifier = Classifier()
#classifier.fit(train_galaxies)

#%%
spiral_ba = BayesianApproximation2d(spiral_pdf)
spiral_ba.run()

plot_ba_2d_results2(spiral_ba)

#%%
elliptic_ba = BayesianApproximation2d(elliptic_pdf)
elliptic_ba.run()

plot_ba_2d_results2(elliptic_ba)

#%%
import statsmodels.api as sm

def plot_ba_2d_results2(ba, size=150000):
    q, xi, zeta, theta, phi = ba.sample(size)
    
    q_kde = sm.nonparametric.KDEUnivariate(q)
    q_kde.fit(bw=0.03)
    q_pdf = q_kde.evaluate(ba.q_pdf.x)
    
    plt.xlim((0, 1))
    plt.plot(ba.q_pdf.x, ba.q_pdf.y, "o", label="$\\rho(q)$")
    plt.plot(ba.q_pdf.x, q_pdf, label="$\\rho(q)$")
    
    plt.legend()

plot_ba_2d_results2(spiral_ba)

#%%
plot_xz_kde(spiral_ba, False)
savefig("plots/xi_zeta_spiral.pdf")

#%%
plot_xz_kde(elliptic_ba, False)
savefig("plots/xi_zeta_elliptic.pdf")

#%%
pos, inc = ba.sample_pos_inc(0.1, 0.5)

#plt.hist(np.abs(pos), 100, label="pos", histtype="step")
plt.hist(np.abs(inc), 100, (0, 1), label="inc", histtype="step")

plt.legend()

#%%
q = 1
bw = 0.005

q_sample, xi, zeta, theta, phi = ba.sample(1000000)
sample = (q_sample > (q - bw)) & (q_sample < (q + bw))

#plt.hist(xi[sample], 100, label="xi", histtype="step")
#plt.hist(zeta[sample], 100, label="zeta", histtype="step")
#plt.hist(np.abs(np.cos(theta[sample])), 100, label="theta", histtype="step")
plt.hist(np.abs(phi[sample] / np.pi), 100, label="phi", histtype="step")

plt.legend()

#%%
from lib.binney import get_q

plt.hist(get_q(xi2, zeta2, theta2, phi2), 100, histtype="step", label="2")
plt.hist(get_q(xi[sample], zeta[sample], theta[sample], phi[sample]), 100, histtype="step", label="1")
plt.legend()

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
