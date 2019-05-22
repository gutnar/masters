#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import dum_bins, galaxy_classes
from src.tex_plot import savefig

#%%
results = {
    "random": ["data/final/random_quantiles.csv", 2, "Juhuslik"],
    "pos": ["data/final/pos_quantiles.csv", 2, "Positsiooninurk"]
}

#%%
for index, (method, method_results) in enumerate(results.items()):
    table = pd.read_csv(method_results[0])

    for c in range(len(galaxy_classes)):
        plt.figure(c)
        plt.ylim((0.85, 1.15))

        plt.plot(
            table["dum_mean"],
            table["%s_low" % galaxy_classes[c]["label"]]
        )

        plt.plot(
            table["dum_mean"],
            table["%s_mean" % galaxy_classes[c]["label"]]
        )

        plt.plot(
            table["dum_mean"],
            table["%s_high" % galaxy_classes[c]["label"]]
        )

#for c in range(len(galaxy_classes)):
#    plt.figure(c)
#    plt.gca().legend(frameon=False)
#    savefig("plots/%s_results.png" % galaxy_classes[c]["label"])
