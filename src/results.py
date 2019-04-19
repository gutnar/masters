#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib import get_dum

#%%
plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams["font.size"] = 16

#%%
f = 0

for m in ("spiral", "elliptic"):
    results = {
        "random": pd.read_csv("data/final/%s_random.csv" % m),
        "global": pd.read_csv("data/final/%s_global.csv" % m),
        "classifier": pd.read_csv("data/final/%s_classifier.csv" % m)
    }

    """
    galaxies = pd.read_csv("data/intermediate/%s.csv" % m)

    results["ba"] = pd.DataFrame({
        "max": np.linspace(0, 1, 100),
        "N": np.histogram(np.concatenate(get_dum(
            galaxies["ra"],
            galaxies["dec"],
            galaxies["pos"] / 180 * np.pi,
            np.arccos(galaxies["ba"]),
            galaxies["gama"],
            galaxies["ex"],
            galaxies["ey"],
            galaxies["ez"]
        )), 100, (0, 1), True)[0] / 100
    })
    """

    for method in results:
        plt.figure(f*2 + 1)
        plt.title("%s" % m)
        plt.plot(
            results[method]["max"],
            results[method]["N"],
            label=method
        )
        plt.gca().legend()

        plt.figure(f*2 + 2)
        plt.title("%s relative to random results" % m)
        plt.plot(
            results[method]["max"],
            results[method]["N"] / results["random"]["N"],
            label=method
        )
        plt.gca().legend()

        #if method != "random":
        #    plt.figure(f*3 + 3)
        #    plt.title("DUM relative to global method results")
        #    plt.plot(
        #        results[method]["max"],
        #        results[method]["N"] / results["global"]["N"],
        #        label=method
        #    )
        #    plt.gca().legend()

    f += 1


#%%
rel = (results["classifier"]["N"] - results["global"]["N"]) / results["global"]["N"]

np.trapz(rel[:50]), np.trapz(rel[50:])
