#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
for method in ("random", "global", "classifier"):
    results = pd.read_csv("data/final/filament_galaxies_%s.csv" % method)
    plt.plot(results["max"], results["N"], label=method)

plt.legend()
