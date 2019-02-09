from inclination import get_inclination
from helpers import get_truncnorm_pdf
from time import time, sleep
from multiprocessing import Pool
from os import cpu_count
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def process_galaxy(galaxy, clf):
    hist = clf.predict_proba([galaxy.values])
    inclination = get_inclination(hist)

    return pd.Series({
        "x_mu": inclination.x_mean,
        "x_sigma": inclination.x_dev,
        "z_mu": inclination.z_mean,
        "z_sigma": inclination.z_dev
    })


def process_galaxies(data):
    return data["galaxies"].apply(process_galaxy, axis=1, result_type="reduce", clf=data["clf"])


if __name__ == '__main__':
    from common import np, plt, galaxies, galaxies_train, parameters
    from classifier import clf

    sample = galaxies[:5000]
    print(sample.describe())

    processes = cpu_count() - 1
    chunks = np.array_split(sample, processes)

    start = time()
    pool = Pool(processes)
    inclinations = pd.concat(pool.map(process_galaxies, [{ "clf": clf, "galaxies": chunk[parameters] } for chunk in chunks]))
    print(time() - start)

    data = pd.concat([sample, inclinations], axis=1, sort=False)
    data.to_csv("data_inclinations.txt", index=False)
