#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

from lib import BayesianApproximation, Classifier
from lib.plotting import *

#%% Load data
test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")
train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")

#%% Train classifier
classifier = Classifier()
classifier.fit(train_galaxies)

#%%
elliptic_galaxies = pd.read_csv("data/intermediate/elliptic_galaxies.csv")
q_pdf = classifier.predict_pdf(test_galaxies.iloc[[0]])
plt.plot(q_pdf.x, q_pdf.y)

#%%
start = time()
ba = BayesianApproximation(q_pdf)
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
t_pdf = ba.get_t_pdf(0.8)
plt.plot(t_pdf.x, t_pdf.y, label=r"$\theta$")
plt.legend()

#%%
p_pdf = ba.get_p_pdf(0.8)
plt.plot(p_pdf.x, p_pdf.y, label=r"$\phi$")
plt.hist(p_pdf.sample(100000), 100, (-np.pi/2, np.pi/2), True)
plt.legend()
