#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lib import BayesianApproximation, Classifier
from lib.plotting import *

#%% Load data
test_galaxies = pd.read_csv("data/intermediate/test_galaxies.csv")
train_galaxies = pd.read_csv("data/intermediate/train_galaxies.csv")

#%% Train classifier
classifier = Classifier(100)
classifier.fit(train_galaxies)

#%%
q_pdf = classifier.predict_pdf(test_galaxies.iloc[[1]])
plt.plot(q_pdf.x, q_pdf.y)

#%%
from time import time

start = time()
ba = BayesianApproximation(q_pdf)
ba.run([(150000, 0.05)]*25)
print(time() - start)

plot_ba_results(ba)

#%%
plot_xz_kde(ba)
