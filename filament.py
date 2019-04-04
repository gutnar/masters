#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from time import time
from multiprocessing import Pool
from os import cpu_count

from helpers import get_truncnorm_sample, PDF
from analytical import get_dum
from bayes import get_xz_kde, sample_cos_t
from classifier import clf, parameters

warnings.filterwarnings("ignore")

#%%

#q_kde = sm.nonparametric.KDEUnivariate(inclinations["ba"].values)
#q_kde.fit(bw=0.01)
q_grid = np.linspace(0, 1, 100)
#q_pdf = PDF(q_kde.evaluate(q_grid), q_grid)

#plt.hist(inclinations["ba"].values, 100, (0, 1), density=True)
#q_pdf.plot()

#%%
#xz_kde = get_xz_kde(q_pdf, True)

#%%
def process_galaxies(galaxies):
    pdfs = clf.predict_proba(galaxies[parameters].values) / 0.01
    cos_t_samples = []
    index = 0

    for i, galaxy in galaxies.iterrows():
        xz_kde = get_xz_kde(PDF(pdfs[index], q_grid), False, 10, 0, 1)
        cos_t = sample_cos_t(galaxy["ba"], xz_kde, 10)
        cos_t_samples.append(cos_t)
        
        index += 1

    return cos_t_samples

#%%
if __name__ == "__main__":
    gama = pd.read_csv("data/raw/gama_data_for_gutnar.txt", r"\s+")
    inclinations = pd.read_csv("data/raw/data_gama_gal_orient.txt", r"\s+")
    galaxies = gama.merge(inclinations, on="id", how="inner")

    #galaxies = galaxies[:120]

    processes = 6

    #pdfs = clf.predict_proba(galaxies[parameters].values) / 0.01
    chunks = np.array_split(galaxies, processes)
    
    start = time()
    pool = Pool(processes)
    cos_t_samples = sum(pool.map(process_galaxies, chunks), [])
    print(time() - start, "estimate_inclination")

    dum = np.array([])
    dum_simple = np.array([])
    dum_random = np.array([])
    index = 0

    for i, galaxy in galaxies.iterrows():
        cos_t = cos_t_samples[index]#sample_cos_t(galaxy["ba"], xz_kde, N)
        index += 1

        dum = np.concatenate((
            dum, get_dum(
                galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
                galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
            )
        ), axis=None)

        # Simple flatness estimation
        dum_simple = np.concatenate((
            dum_simple, get_dum(
                galaxy["ra"], galaxy["dec"], galaxy["pos"], (float(galaxy["ba"]), ),
                galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
            )
        ))

        # Random inclination angle
        cos_t = np.random.uniform(0, 1, 10)

        dum_random = np.concatenate((
            dum_random, get_dum(
                galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
                galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
            )
        ))
    
    kde = sm.nonparametric.KDEUnivariate(dum)
    kde_simple = sm.nonparametric.KDEUnivariate(dum_simple)
    kde_random = sm.nonparametric.KDEUnivariate(dum_random)

    kde.fit(bw=0.05, cut=0)
    kde_simple.fit(bw=0.05, cut=0)
    kde_random.fit(bw=0.05, cut=0)

    plt.figure(1)
    plt.hist(galaxies["ba"].values, 100, (0, 1), True, histtype="step")
    plt.hist(np.array(cos_t_samples).flatten(), 100, (0, 1), True, histtype="step")
    plt.savefig("plots/cos.png")

    plt.figure(2)
    plt.plot(kde_random.support, kde_random.density, label="random")
    plt.plot(kde_simple.support, kde_simple.density, label="b/a")
    plt.plot(kde.support, kde.density, label="bayes")
    plt.legend()

    plt.savefig("plots/dum.png")


'''
#%%
cos_t_samples = np.array([])

for i, galaxy in galaxies.iterrows():
    cos_t = sample_cos_t(galaxy["ba"], xz_kde, 100)

    cos_t_samples = np.concatenate((
        cos_t_samples,
        cos_t
    ))

plt.hist(galaxies["ba"].values, 100, (0, 1), True, histtype="step")
plt.hist(cos_t_samples, 100, (0, 1), True, histtype="step")

#%%
N = 100
dum = np.array([])
dum_simple = np.array([])
dum_random = np.array([])

for index, galaxy in galaxies.iterrows():
    cos_t = sample_cos_t(galaxy["ba"], xz_kde, N)

    dum = np.concatenate((
        dum, get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ), axis=None)

    # Simple flatness estimation
    dum_simple = np.concatenate((
        dum_simple, get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["pos"], (float(galaxy["ba"]), ),
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ))

    # Random inclination angle
    cos_t = np.random.uniform(0, 1, N)

    dum_random = np.concatenate((
        dum_random, get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ))

#%%
kde = sm.nonparametric.KDEUnivariate(dum)
kde_simple = sm.nonparametric.KDEUnivariate(dum_simple)
kde_random = sm.nonparametric.KDEUnivariate(dum_random)

kde.fit(bw=0.05, cut=0)
kde_simple.fit(bw=0.05, cut=0)
kde_random.fit(bw=0.05, cut=0)

plt.plot(kde_random.support, kde_random.density, label="random")
plt.plot(kde_simple.support, kde_simple.density, label="b/a")
plt.plot(kde.support, kde.density, label="bayes")
plt.legend()

plt.savefig("plots/dum.png")
'''
