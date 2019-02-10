#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import get_truncnorm_sample

N = 1000000

#x_mu       0.313408
#x_sigma    0.096828
#z_mu       0.962000
#z_sigma    0.043300
#dtype: float64

#x = np.random.normal(0.14, 0.1, N)
#z = np.random.normal(0.8, 0.05, N)

x = get_truncnorm_sample(0.22, 0.29, 0, 1, N)
z = get_truncnorm_sample(0.96, 0.09, 0, 1, N)

x = np.minimum(x, z)
z = np.maximum(x, z)

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

#%%
plt.hist(ba, np.linspace(0, 1, 100))

df = pd.DataFrame({
    "x": x,
    "z": z,
    "cos_t": cos_t,
    "cos_p": cos_p,
    "ba": ba
})

df.to_csv("data_generated.txt", index=False)

#%%
from time import time

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
    bahist = np.histogram(ba, 100, (0, 1), density=True)[0]

    return bahist

#%%
columns = ["x", "z"] + ["ba" + str(i) for i in range(100)]
df = pd.DataFrame([], columns=columns)
X_step = 0.005
Z_step = 0.005

for X in np.linspace(0, 0.5, 100, endpoint=False):
    for Z in np.linspace(0.75, 1, 50, endpoint=False):
        x_values = np.random.uniform(X, X + X_step, 10)
        z_values = np.random.uniform(Z, Z + Z_step, 10)

        for i in range(10):
            x = np.repeat(x_values[i], 100)
            z = np.repeat(z_values[i], 100)
            ba_hist = get_ba_hist(x, z, 100)

            df = df.append(
                pd.DataFrame(
                    [np.concatenate((
                        np.array([x_values[i], z_values[i]]), ba_hist
                    ))],
                    columns=columns
                )
            )

df.describe()

#%%
np.linspace(0.75, 1, 500, endpoint=False) #0.005