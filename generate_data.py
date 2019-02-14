#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import get_truncnorm_sample, get_ba
from sklearn.ensemble import RandomForestClassifier
from time import time
from multiprocessing import Pool
from os import cpu_count

#%%
ba_columns = ["ba" + str(i) for i in range(100)]
columns = ["x_slot", "z_slot"] + ba_columns
X_bins = np.linspace(0, 0.4, 80 + 1)
Z_bins = np.linspace(0.8, 1, 40 + 1)

def generate(N):
    df = pd.DataFrame([], columns=columns)

    for X in range(len(X_bins) - 1):
        for Z in range(len(Z_bins) - 1):
            x = np.random.uniform(X_bins[X], X_bins[X+1], N)
            z = np.random.uniform(Z_bins[Z], Z_bins[Z+1], N)
            ba = get_ba(x, z)
            ba_hist = np.histogram(ba, 100, (0, 1), density=True)[0]

            df = df.append(
                pd.DataFrame(
                    [np.concatenate((
                        np.array([X, Z]), ba_hist
                    ))],
                    columns=columns
                )
            )
    
    return df

#%%
if __name__ == '__main__':
    start = time()
    pool = Pool(cpu_count() - 1)
    df = pd.concat(pool.map(generate, np.repeat(1000, 20)))
    print(time() - start)

    df["x_slot"] = df["x_slot"].astype(int)
    df["z_slot"] = df["z_slot"].astype(int)
    
    print(df.describe())
    df.to_csv("data_generated.txt", index=False)

#%%
'''
clf = RandomForestClassifier(
    n_estimators=64,
    #max_depth=10,
    #min_samples_split=5,
    min_samples_leaf=2,
    max_features=None,
    n_jobs=-1
)

df["x_slot"] = df["x_slot"].apply(np.floor)

X_train = df[ba_columns].values
Y_train = df["x_slot"].values

clf.fit(X_train, Y_train)

#%%
#clf.predict_proba(df.iloc[[5000]][ba_columns].values)
#df.iloc[[0]][ba_columns].values
#df.iloc[[5000]]["x_slot"]

#%%
galaxies = pd.read_csv("data_gama_gal_orient.txt", sep="\s+")
#galaxies["baslot"] = galaxies["ba"].multiply(100).apply(np.ceil) - 1
#plt.hist(galaxies["baslot"], 100, (0, 100), density=True)

hist = np.histogram(galaxies["ba"], 100, (0, 1), density=True)[0]
probs = clf.predict_proba([hist])

plt.plot(probs[0])
'''

#%%
h = np.histogram([0.1, 0.1, 0.2, 0.3, 0.4], 4, (0, 1))

h[0] / 5
