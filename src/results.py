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
plt.ylim((0.6, 1.4))
plot_result_1("random", 0)
plot_result_1("spiral_pos", 0)

#%%
plt.ylim((0.6, 1.4))
plot_result_1("random", 1)
plot_result_1("elliptic_pos", 1)

#%% global spiral results
plt.ylim((0.6, 1.4))
plot_result_1("random", 0)
plot_result_1("spiral_pos", 0)
plot_result_1("global", 0)

#%% global elliptic results
plt.ylim((0.6, 1.4))
plot_result_1("random", 1)
plot_result_1("elliptic_pos", 1)
plot_result_1("global", 1)

#%% rf spiral results
plt.ylim((0.9, 1.1))
plot_result_1("random", 0)
plot_result_1("rf", 0)

#%% rf elliptic results
plt.ylim((0.9, 1.1))
plot_result_1("random", 1)
plot_result_1("rf", 1)
