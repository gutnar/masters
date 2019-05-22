#%%
import pandas as pd
import matplotlib.pyplot as plt

from lib import PDF, Classifier, BayesianApproximation2d
from lib.plotting import *
from src.tex_plot import savefig

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
classifier = Classifier(25, n_estimators=6, max_depth=10, max_features=2, bootstrap=True, criterion="entropy")
classifier.fit(train_galaxies)

#%%
predicted_pdf = classifier.predict_pdf(test_galaxies.iloc[[12354]])
plt.plot(predicted_pdf.x, predicted_pdf.y)

#%%
ba = BayesianApproximation2d(predicted_pdf, 150000)
ba.run([(150000, "scott")]*50)# + [(150000, "scott")]*25)

plot_ba_2d_results(ba)

#%%
1/150000**(1/6)

#%%
plot_xz_kde(ba)

plt.tick_params(direction="in")
#plt.savefig("plots/xi_zeta_intial_kde.pdf", dpi=1000, bbox_inches='tight')#, pad_inches=0)

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
