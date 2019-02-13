#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import get_truncnorm_sample
from sklearn.ensemble import RandomForestClassifier
from time import time
from multiprocessing import Pool
from os import cpu_count


def get_ba_hist(x, z, N):
    x2 = x**2
    z2 = z**2

    cos_t = np.random.uniform(0, 1, N)
    cos2_t = cos_t**2
    sin2_t = 1 - cos2_t

    cos_p = np.random.uniform(0, 1, N)
    cos2_p = cos_p**2
    sin2_p = 1 - cos2_p
    sin_2p = 2*np.sqrt(sin2_p)*np.sqrt(cos2_p)

    A = cos2_t/x2 * (sin2_p + cos2_p/z2) + sin2_t/z2
    B = (1 - 1/z2) * 1/x2 * cos_t * sin_2p
    C = (sin2_p/z2 + cos2_p)/x2
    D = np.sqrt((A - C)**2 + B**2)

    ba = np.sqrt((A + C - D) / (A + C + D))
    #baslot = np.ceil(ba * 100) - 1
    ba_hist = np.histogram(ba, 100, (0, 1))[0] / N

    return ba_hist

#%%
ba_columns = ["ba" + str(i) for i in range(100)]
columns = ["x_slot", "z_slot"] + ba_columns
X_step = 0.005
Z_step = 0.005

def generate(N):
    df = pd.DataFrame([], columns=columns)

    for X in np.linspace(0, 0.5, 100, endpoint=False):
        for Z in np.linspace(0.75, 1, 50, endpoint=False):
            x = np.random.uniform(X, X + X_step, N)
            z = np.random.uniform(Z, Z + Z_step, N)
            ba_hist = get_ba_hist(x, z, N)

            df = df.append(
                pd.DataFrame(
                    [np.concatenate((
                        np.array([int(X/X_step), int((Z-0.75)/Z_step)]), ba_hist
                    ))],
                    columns=columns
                )
            )
    
    return df

#%%
if __name__ == '__main__':
    start = time()
    pool = Pool(cpu_count() - 1)
    df = pd.concat(pool.map(generate, np.repeat(1000, 10)))
    print(time() - start)
    
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
