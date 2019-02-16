#%%
from inclination import estimate_inclination
from helpers import get_truncnorm_pdf
from time import time, sleep
from multiprocessing import Pool
from os import cpu_count
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def process_galaxy(galaxy, clf):
    pdf = clf.predict_proba([galaxy.values])[0] / 0.01
    result = estimate_inclination(pdf)

    print(galaxy.name)

    return pd.Series({
        "x_mu": result[0],
        "x_sigma": result[1],
        "z_mu": result[2],
        "z_sigma": result[3]
    })


def process_galaxies(pdfs):
    return [estimate_inclination(pdf) for pdf in pdfs]
    #pdfs = data["clf"].predict_proba(data["galaxies"].values) / 0.01
    #return data["galaxies"].apply(process_galaxy, axis=1, result_type="reduce", clf=data["clf"])


if __name__ == '__main__':
    from classifier import galaxies, clf, parameters

    sample = galaxies[:50]
    print(sample.describe())

    processes = 4#cpu_count() - 1

    start = time()
    pdfs = clf.predict_proba(sample[parameters].values)
    #chunks = np.array_split(sample, processes)
    chunks = np.array_split(pdfs, processes)
    print(time() - start, "predict_proba")

    start = time()
    pool = Pool(processes)
    #inclinations = pool.map(process_galaxies, [{
    #    "clf": clf, "galaxies": chunk[parameters]
    #} for chunk in chunks])
    inclinations = sum(pool.map(process_galaxies, chunks), [])
    print(time() - start, "estimate_inclination")

    data = pd.concat([
        sample, pd.DataFrame.from_records(inclinations, columns=("x_mu", "x_sigma", "z_mu", "z_sigma"))
    ], axis=1, sort=False)

    data.to_csv("data_inclinations.txt", index=False)
