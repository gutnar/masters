#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams["font.size"] = 16

#%%
random_results = pd.read_csv("data/final/filament_galaxies_random.csv")

for method in ("random", "global", "classifier"):
    results = pd.read_csv("data/final/filament_galaxies_%s.csv" % method)
    
    plt.figure(1)
    plt.plot(
        results["max"],
        results["N"],
        label=method
    )

    plt.figure(2)
    plt.plot(
        results["max"],
        (results["N"] - random_results["N"]) / random_results["N"],
        label=method
    )

plt.figure(1)
plt.gca().legend()
plt.savefig("plots/dum_absolute.png")

plt.figure(2)
plt.gca().legend()
plt.savefig("plots/dum_relative.png")
