#%%
from inclination import get_inclination
from helpers import get_truncnorm_pdf
from time import time, sleep
import pandas as pd
from common import np, plt, galaxies, galaxies_train, parameters
from classifier import clf


hist = np.histogram(galaxies["ba"], 100, (0, 100), density=True)[0]
inclination = get_inclination(hist)

pd.Series({
    "x_mu": inclination.x_mean,
    "x_sigma": inclination.x_dev,
    "z_mu": inclination.z_mean,
    "z_sigma": inclination.z_dev
})

#x_mu       0.313408
#x_sigma    0.096828
#z_mu       0.962000
#z_sigma    0.043300
