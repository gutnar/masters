#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from plot_cos_t import plot_cos_t
from inclination import sample_cos_t
from helpers import get_truncnorm_sample
import warnings

warnings.filterwarnings("ignore")

#%%
gama = pd.read_csv("data/raw/gama_data_for_gutnar.txt", sep=r"\s+")
inclinations = pd.read_csv("data/intermediate/inclinations.txt")

galaxies = gama.merge(inclinations, on="id", how="inner")
plot_cos_t(galaxies)

#%%
def calc_gal_spin_vec(ra, dec, pos, inc):
    ra_buf = ra * np.pi / 180
    dec_buf = dec * np.pi / 180
    pos_buf = (pos * np.pi / 180 + np.pi) % np.pi
    inc_buf = inc * np.pi / 180

    # Find the components of unit spin vectors in the local
    # topocentric coordinate frame.
    u = -np.sin(inc_buf) * np.cos(pos_buf)
    v = np.sin(inc_buf) * np.sin(pos_buf)
    w = np.cos(inc_buf) # pointing away from Earth

    # Compute to alternate endpoints if the vectors were shifted
    # to the origin of the GEI frame.

    # The topocentric elevation component pointing away from Earth.
    dvec1 = np.transpose(np.array([
        -u*np.sin(ra_buf) - v*np.sin(dec_buf) * np.cos(ra_buf) + w*np.cos(dec_buf)*np.cos(ra_buf),
        u*np.cos(ra_buf) - v*np.sin(dec_buf)*np.sin(ra_buf) + w*np.cos(dec_buf)*np.sin(ra_buf),
        v*np.cos(dec_buf) + w*np.sin(dec_buf)
    ]))

    # The topocentric elevation component pointing towards Earth
    dvec2 = np.transpose(np.array([
        -u*np.sin(ra_buf) - v*np.sin(dec_buf)*np.cos(ra_buf) - w*np.cos(dec_buf)*np.cos(ra_buf),
        u*np.cos(ra_buf) - v*np.sin(dec_buf)*np.sin(ra_buf) - w*np.cos(dec_buf)*np.sin(ra_buf),
        v*np.cos(dec_buf) - w*np.sin(dec_buf)
    ]))

    # Safety procedure
    dvec1 = normalize(dvec1, axis=1)
    dvec2 = normalize(dvec2, axis=1)

    return dvec1, dvec2

#%%
def get_crd(ra, dec, dist):
    rpi = np.pi/180

    return np.array([
        dist * np.cos(ra*rpi) * np.cos(dec*rpi),
        dist * np.sin(ra*rpi) * np.cos(dec*rpi),
        dist * np.sin(dec*rpi)
    ])

#%%
def get_rotated_gama_coordinates_crd(crd0, gama):
    if gama == 9:
        ra0 = 135.0
        dec0 = 0.5
    elif gama == 12:
        ra0 = 180.0
        dec0 = -0.5
    elif gama == 15:
        ra0 = 217.5
        dec0 = 0.5
    
    crd = get_crd(ra0, dec0, 1.0)
    los = crd/np.sqrt(np.sum(crd**2)) # line of sight vector
    crd = get_crd(ra0, dec0 - 1.0, 100.0)
    crd2 = get_crd(ra0, dec0 + 1.0, 100.0)
    crd = crd2 - crd
    dum = np.vdot(crd, los) # crd projektsioon los suunal
    crd2 = dum * los
    crd = crd - crd2
    ex = crd/np.sqrt(np.sum(crd**2)) # vaatekiirega ristuv vektor
    ey = np.array([
        los[1] * ex[2] - los[2] * ex[1],
        los[2] * ex[0] - los[0] * ex[2],
        los[0] * ex[1] - los[1] * ex[0]
    ])
    ey = ey/np.sqrt(np.sum(ey**2))
    
	# risti vektorid on los, ey,ey
	# projekteerime koordinaadid ümber
    return np.dot(crd0, np.transpose(np.array([
        ex, ey, los
    ])))

#%%
def get_dum(ra, dec, pos, cos_t, gama, ex, ey, ez):
    dvec1, dvec2 = calc_gal_spin_vec(
        ra, dec, pos, np.arccos(cos_t) / np.pi * 180
    )

    dvec = np.concatenate((dvec1, dvec2))
    crd = get_rotated_gama_coordinates_crd(dvec, gama)

    fil = np.transpose(np.array([
        ex, ey, ez
    ]))

    return np.abs(np.dot(crd, fil))

#%%
N = 100
dum = np.array([])
dum_simple = np.array([])
dum_random = np.array([])

for index, galaxy in galaxies.iterrows():
    cos_t = sample_cos_t(
        galaxy["ba"],
        galaxy["x_mu"], galaxy["x_sigma"],
        galaxy["z_mu"], galaxy["z_sigma"],
        N
    )

    cos_t = cos_t[~np.isnan(cos_t)]

    if len(cos_t) == 0:
        continue

    dum = np.concatenate((
        dum, get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
            galaxy["gama"], galaxy["ex"], galaxy["ey"], galaxy["ez"]
        )
    ), axis=None)

    # Simple flatness estimation
    cos_t = np.repeat(float(galaxy["ba"]), N)

    dum_simple = np.concatenate((
        dum_simple, get_dum(
            galaxy["ra"], galaxy["dec"], galaxy["pos"], cos_t,
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

#%%
plt.plot(kde.support, kde.density, label="Complex")
plt.plot(kde_simple.support, kde_simple.density, label="Simple")
plt.plot(kde_random.support, kde_random.density, label="Random")
plt.legend()

plt.savefig("plots/dum.png")
