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
for m in np.linspace(0, 1, 10):
    xi = np.concatenate((
        np.random.normal(0.222, 0.057, int(10000*(1 - m))),
        np.random.normal(0.7, 0.1, int(10000*m))
    ))

    #xi = []
    zeta = np.random.normal(0.85, 0.1, len(xi))

    #for i in range(int(10000*(1 - m))):
    #    xi.append(get_truncnorm_sample(0.222, 0.057, 0, zeta[i], 1)[0])
    
    #for i in range(len(xi), len(zeta)):
    #    xi.append(get_truncnorm_sample(0.7, 0.1, 0, zeta[i], 1)[0])

    xz_kde = stats.kde.gaussian_kde(np.column_stack((np.array(xi), zeta)).T)

    ba = BayesianApproximation2d(predicted_pdf, xz_kde)
    #ba.run([(1000, "scott")]*25)

    print(m, ba.error(1000))

plot_ba_2d_results(ba)

#%%
ba = BayesianApproximation2d(predicted_pdf)
ba.run()

plot_ba_2d_results(ba)

#%%
plot_xz_kde(ba)
#plt.tick_params(direction="in")
#plt.savefig("plots/xi_zeta_intial_kde.pdf", dpi=1000, bbox_inches='tight')#, pad_inches=0)

#%%
pos, inc = ba.sample_pos_inc(0.9, 0, 10000)

plt.hist(pos, 100, label="pos")
plt.hist(inc, 100, label="inc")

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
