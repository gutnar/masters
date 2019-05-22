#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import dum_bins, galaxy_classes
from src.tex_plot import savefig

#%%
results = {
    "random": ["data/final/random_quantiles.csv", 2, "Juhuslik"],
    "pos": ["data/final/pos_quantiles.csv", 2, "Positsiooninurk ja f"],
    #"spiral_pos": ["data/final/spiral_pos_quantiles.csv", 2, "Spiraalne f"],
    #"elliptic_pos": ["data/final/spiral_pos_quantiles.csv", 2, "Elliptiline f"],
    "global": ["data/final/global_quantiles.csv", 2, "Global"]
}

#%%
def plot_result(method, c):
    table = pd.read_csv("data/final/%s_quantiles.csv" % method)
    galaxy_class = galaxy_classes[c]

    #plt.ylim((0.75, 1.25))

    plt.plot(
        table["dum_mean"],
        table["%s_mean" % galaxy_class["label"]]
    )

    plt.fill_between(
        table["dum_mean"],
        table["%s_low" % galaxy_class["label"]],
        table["%s_high" % galaxy_class["label"]],
        alpha=.5
    )

#%%
plot_result("random", 0)
#plot_result("random", 1)

#%% spiral results
plot_result("random", 0)
plot_result("spiral_pos", 0)
plot_result("global", 0)

#%% elliptic results
plot_result("random", 0)
plot_result("elliptic_pos", 1)
plot_result("global", 1)

#%% spiral

#%%
#for c in range(len(galaxy_classes)):
#    plt.figure(c)
#    plt.gca().legend(frameon=False)
#    savefig("plots/%s_results.png" % galaxy_classes[c]["label"])
