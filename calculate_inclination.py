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


def process_galaxies(pdfs):
    return [estimate_inclination(pdf) for pdf in pdfs]


if __name__ == '__main__':
    from classifier import galaxies, clf, parameters

    sample = galaxies
    print(sample.describe())

    processes = 10#cpu_count() - 1

    start = time()
    pdfs = clf.predict_proba(sample[parameters].values) / 0.01
    chunks = np.array_split(pdfs, processes)
    print(time() - start, "predict_proba")

    start = time()
    pool = Pool(processes)
    inclinations = sum(pool.map(process_galaxies, chunks), [])
    print(time() - start, "estimate_inclination")

    data = pd.concat([
        sample, pd.DataFrame.from_records(inclinations, columns=("x_mu", "x_sigma", "z_mu", "z_sigma"))
    ], axis=1, sort=False)

    data.to_csv("data_inclinations.txt", index=False)
