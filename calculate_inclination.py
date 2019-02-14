#%%
from inclination import estimate_inclination
from helpers import get_truncnorm_pdf
from time import time, sleep
from multiprocessing import Pool
from os import cpu_count
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def process_galaxy(galaxy, clf):
    pdf = np.concatenate([clf.predict_proba([galaxy.values]), np.array([0])]) / 0.005
    result = estimate_inclination(pdf)

    return pd.Series({
        "x_mu": result[0],
        "x_sigma": result[1],
        "z_mu": result[2],
        "z_sigma": result[3]
    })


def process_galaxies(data):
    return data["galaxies"].apply(process_galaxy, axis=1, result_type="reduce", clf=data["clf"])


if __name__ == '__main__':
    from common import np, plt, galaxies, galaxies_train, parameters
    from classifier import clf

    sample = galaxies[:100]
    print(sample.describe())

    processes = cpu_count() - 1
    chunks = np.array_split(sample, processes)

    start = time()
    pool = Pool(processes)
    inclinations = pd.concat(pool.map(process_galaxies, [{ "clf": clf, "galaxies": chunk[parameters] } for chunk in chunks]))
    print(time() - start)

    data = pd.concat([sample, inclinations], axis=1, sort=False)
    data.to_csv("data_inclinations.txt", index=False)
