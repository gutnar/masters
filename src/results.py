#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams["font.size"] = 16

#%%
results = {
    "random": pd.read_csv("data/final/filament_galaxies_random.csv"),
    "global": pd.read_csv("data/final/filament_galaxies_global.csv"),
    "classifier": pd.read_csv("data/final/filament_galaxies_classifier.csv")
}

for method in results:
    plt.figure(1)
    plt.plot(
        results[method]["max"],
        results[method]["N"],
        label=method
    )

    plt.figure(2)
    plt.plot(
        results[method]["max"],
        (results[method]["N"] - results["random"]["N"]) / results["random"]["N"],
        label=method
    )

    if method != "random":
        plt.figure(3)
        plt.plot(
            results[method]["max"],
            (results[method]["N"] - results["global"]["N"]) / results["global"]["N"],
            label=method
        )

plt.figure(1)
plt.title("DUM")
plt.gca().legend()
plt.savefig("plots/dum_absolute.png")

plt.figure(2)
plt.title("DUM relative to random results")
plt.gca().legend()
plt.savefig("plots/dum_relative_to_random.png")

plt.figure(3)
plt.title("DUM relative to global method results")
plt.gca().legend()
plt.savefig("plots/dum_relative_to_global.png")

#%%
rel = (results["classifier"]["N"] - results["global"]["N"]) / results["global"]["N"]

np.trapz(rel[:50]), np.trapz(rel[50:])
