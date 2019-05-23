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
filament_galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")
spiral_galaxies = filament_galaxies[filament_galaxies["e_class"] == 0]
elliptic_galaxies = filament_galaxies[filament_galaxies["e_class"] == 1]

#predicted_pdf = classifier.predict_pdf(elliptic_galaxies.iloc[[2]])
predicted_pdf = PDF.from_samples(
    np.linspace(0, 1, 100),
    spiral_galaxies["ba"].values
)

plt.plot(predicted_pdf.x, predicted_pdf.y)

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
pos, inc = ba.sample_pos_inc(0.9, 0, 5000, 0.001)

plt.hist(pos, 100, label="pos")
plt.hist(inc, 100, label="inc")

plt.legend()

#%%
q = 0.95
bw = 0.001

q_sample, xi, zeta, theta, phi = ba.sample(1000000)
sample = (q_sample > (q - bw)) & (q_sample < (q + bw)) & (zeta > 0.5)

kde = stats.kde.gaussian_kde(np.column_stack((
    xi, zeta, theta, phi
    #xi[sample], zeta[sample], theta[sample], phi[sample]
)).T)

xi, zeta, theta, phi = kde.resample(5000)
#xi[xi < 0] = -xi[xi < 0]
#zeta[zeta > 1] = 2 - zeta[zeta > 1]

#xi[xi > zeta] = 2*zeta[xi > zeta] - xi[xi > zeta]
#xi[xi > 1] = 2 - xi[xi > 1]
#zeta[zeta < 0] = -zeta[zeta < 0]

#plt.hist(xi, 100, label="xi", histtype="step")
#plt.hist(zeta, 100, label="zeta", histtype="step")
plt.hist(np.abs(np.cos(theta)), 100, label="theta", histtype="step")
#plt.hist(phi, 100, label="phi", histtype="step")
plt.legend()

#%%
q = 0.5
bw = 0.005

q_sample, xi, zeta, theta, phi = ba.sample(1000000)
sample = (q_sample > (q - bw)) & (q_sample < (q + bw)) & (zeta > 0.5)

kde = stats.kde.gaussian_kde(np.column_stack((
    #xi, zeta, theta, phi
    xi[sample], zeta[sample], theta[sample], phi[sample]
)).T, 0.025)

xi2, zeta2, theta2, phi2 = kde.resample(5000)
#xi2[xi2 < 0] = -xi2[xi2 < 0]

plt.hist(np.abs(np.cos(theta2)), 100, label="theta", histtype="step")
#plt.hist(theta2/np.pi, 100, label="theta", histtype="step")
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
