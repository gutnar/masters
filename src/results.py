#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import galaxy_classes
from src.tex_plot import savefig


def plot_result(method, c, label=""):
    table = pd.read_csv("data/final/%s_quantiles.csv" % method)
    galaxy_class = galaxy_classes[c]

    mean = table["%s_mean" % galaxy_class["label"]]
    std = table["%s_std" % galaxy_class["label"]] * 1.96

    dum = table["dum_mean"].values
    dum[0] = 0
    dum[-1] = 1

    plt.xlim((0, 1))
    plt.xlabel("$|\\vec{n} \cdot \\vec{\\omega}|$")
    plt.ylabel("$\\rho(|\\vec{n} \cdot \\vec{\\omega}|)$", rotation=0, labelpad=25)
    plt.plot(dum, mean, label=label)
    plt.fill_between(dum, mean - std, mean + std, alpha=.5)

#%% global spiral results
plt.ylim((0.6, 1.4))
plot_result("random", 0, "Juhuslikud nurgad")
plot_result("spiral_pos", 0, "Ketta lähendus")
plot_result("sern", 0, "Ellipsoidi lähendus")
plt.legend(frameon=False)
savefig("plots/results_spiral.pdf")

#%% global elliptic results
plt.ylim((0.6, 1.4))
plot_result("random", 1, "Juhuslikud nurgad")
plot_result("elliptic_pos", 1, "Ketta lähendus")
plot_result("sern", 1, "Ellipsoidi lähendus")
plt.legend(frameon=False)
savefig("plots/results_elliptic.pdf")

#%% rf spiral results
plt.ylim((0.9, 1.1))
plot_result("random", 0)
plot_result("rf", 0)

#%% rf elliptic results
plt.ylim((0.9, 1.1))
plot_result("random", 1)
plot_result("rf", 1)
