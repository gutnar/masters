#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import dum_bins, galaxy_classes
from src.tex_plot import savefig

#%%
def plot_result(method, c, label=""):
    table = pd.read_csv("data/final/%s_quantiles.csv" % method)
    galaxy_class = galaxy_classes[c]

    plt.plot(
        table["dum_mean"],
        table["%s_mean" % galaxy_class["label"]],
        label=label
    )

    plt.fill_between(
        table["dum_mean"],
        table["%s_low" % galaxy_class["label"]],
        table["%s_high" % galaxy_class["label"]],
        alpha=.5
    )


def plot_result_1(method, c, label=""):
    table = pd.read_csv("data/final/%s_quantiles.1.csv" % method)
    galaxy_class = galaxy_classes[c]

    mean = table["%s_mean" % galaxy_class["label"]]
    std = table["%s_std" % galaxy_class["label"]] * 1.96

    plt.plot(table["dum_mean"], mean, label=label)
    plt.fill_between(table["dum_mean"], mean - std, mean + std, alpha=.5)

#%%
#plt.ylim((0.9, 1.1))
plot_result_1("random", 0)
plot_result_1("spiral_pos", 0)

#%%
#plt.ylim((0.9, 1.1))
plot_result_1("random", 1)
plot_result_1("elliptic_pos", 1)

#%% spiral results
plt.ylim((0.75, 1.25))
plot_result("random", 0)
plot_result("spiral_pos", 0)

#%% elliptic results
plt.ylim((0.75, 1.25))
plot_result("random", 1)
plot_result("elliptic_pos", 1)

#%% global spiral results
plt.ylim((0.94, 1.06))
plot_result("random", 0, "Juhuslikud nurgad")
plot_result("global", 0, "Kogu valimi $q$ jaotuse järgi")
plt.legend(frameon=False)
savefig("plots/results_global_spiral.pdf")

#%% global elliptic results
plt.ylim((0.94, 1.06))
plot_result("random", 1, "Juhuslikud nurgad")
plot_result("global", 1, "Kogu valimi $q$ jaotuse järgi")
plt.legend(frameon=False)
savefig("plots/results_global_elliptic.pdf")

#%%
#for c in range(len(galaxy_classes)):
#    plt.figure(c)
#    plt.gca().legend(frameon=False)
#    savefig("plots/%s_results.png" % galaxy_classes[c]["label"])
