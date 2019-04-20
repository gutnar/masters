#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib import get_dum

#%%
hist_random = pd.read_csv("data/final/filament_galaxies_random.csv")
hist_global = pd.read_csv("data/final/filament_galaxies_global_0.6.csv")
hist_classifier = pd.read_csv("data/final/filament_galaxies_classifier.csv")

plt.figure(1)
plt.imshow(hist_random.values.T, origin="lower")

plt.figure(2)
plt.imshow(hist_global.values.T, origin="lower")

plt.figure(3)
plt.imshow(hist_classifier.values.T, origin="lower")

#%%
dum_values = (np.linspace(0, 1, 101)[:-1] + np.linspace(0, 1, 101)[1:]) / 2
sern_values = (np.linspace(0, 20, 51)[:-1] + np.linspace(0, 20, 51)[1:]) / 2
sern_bins = [0, 2, 20]

hr = np.zeros((len(sern_bins) - 1, 100))
hg = np.zeros((len(sern_bins) - 1, 100))
hc = np.zeros((len(sern_bins) - 1, 100))

for i in range(len(sern_bins) - 1):
    hr[i] = hist_random[(sern_values >= sern_bins[i]) & (sern_values < sern_bins[i + 1])].sum(axis=0)
    hg[i] = hist_global[(sern_values >= sern_bins[i]) & (sern_values < sern_bins[i + 1])].sum(axis=0)
    hc[i] = hist_classifier[(sern_values >= sern_bins[i]) & (sern_values < sern_bins[i + 1])].sum(axis=0)

plt.plot(sern_bins[1:], np.sum(hr * dum_values, axis=1) / np.sum(hr, axis=1), label="random")
plt.plot(sern_bins[1:], np.sum(hg * dum_values, axis=1) / np.sum(hg, axis=1), label="global")
plt.plot(sern_bins[1:], np.sum(hc * dum_values, axis=1) / np.sum(hc, axis=1), label="classifier")
plt.legend()

#%%
for i in range(len(sern_bins) - 1):
    plt.figure(i)
    plt.title("%d < sern < %d" % (sern_bins[i], sern_bins[i+1]))
    s = hr[i].sum()
    plt.plot(dum_values, hr[i] / s, label="random")
    plt.plot(dum_values, hg[i] / s, label="global")
    plt.plot(dum_values, hc[i] / s, label="classifier")
    plt.gca().legend()

"""
plt.figure(len(sern_bins))
plt.title("0 < sern < 20")
plt.plot(dum_values, hist_random.sum(axis=0), label="random")
plt.plot(dum_values, hist_global.sum(axis=0), label="global")
plt.plot(dum_values, hist_classifier.sum(axis=0), label="classifier")
plt.gca().legend()
"""
