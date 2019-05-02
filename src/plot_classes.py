#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from lib import PDF, BayesianApproximation2d
from lib.plotting import *
from src.common import n_clusters, galaxy_classes

#%%
galaxies = pd.read_csv("data/intermediate/galaxies.csv")

galaxies["e_class"] = np.select(
    (galaxies["sern"] < 2, galaxies["sern"] > 2),
    (0, 1)
)

galaxies.describe()

#%%
for c, galaxy_class in enumerate(galaxy_classes):
    values = galaxies[galaxy_class["parameter"]]
    sample = galaxies[values == galaxy_class["value"]]

    ba = BayesianApproximation2d(PDF.from_samples(
        np.linspace(0, 1, 100),
        sample["ba"].values
    ))
    ba.run()

    plt.figure(c*2)
    plot_ba_2d_results(ba)
    plt.savefig("plots/%s_ba.png" % galaxy_class["label"])

    plt.figure(c*2 + 1)
    plot_xz_kde(ba)
    plt.savefig("plots/%s_xz.png" % galaxy_class["label"])
