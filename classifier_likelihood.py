#%%
from helpers import get_truncnorm_pdf
from time import time, sleep
from scipy.signal import savgol_filter
from time import time
from multiprocessing import Pool
from os import cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")


def process_galaxy(galaxy, clf, parameters, sample_pdf):
    pdf = np.concatenate([clf.predict_proba([galaxy[parameters].values])[0], np.array([0])]) / 0.005
    smooth = savgol_filter(pdf, 31, 3)
    baslot = int(galaxy["baslot"])

    return pd.Series({
        "sample": (sample_pdf[baslot] + smooth[baslot + 1]) / 2 * 0.005,
        "classifier": (smooth[baslot] + smooth[baslot + 1]) / 2 * 0.005
    })


def process_galaxies(data):
    return data[0].apply(process_galaxy, axis=1, result_type="reduce", args=[data[1], data[2], data[3]])


#%%
if __name__ == '__main__':
    from classifier import galaxies_test, clf, parameters
    
    sample = galaxies_test
    print(sample.describe())

    kde = sm.nonparametric.KDEUnivariate(sample["ba"])
    kde.fit()
    sample_pdf = kde.evaluate(np.linspace(0, 1, 101))

    processes = cpu_count() - 1
    chunks = np.array_split(sample, processes)

    start = time()
    pool = Pool(processes)
    p = pd.concat(pool.map(process_galaxies, [(chunk, clf, parameters, sample_pdf) for chunk in chunks]))
    print(time() - start)

    print(p.describe())
    print("sample", np.sum(p["sample"]))
    print("classifier", np.sum(p["classifier"]))
