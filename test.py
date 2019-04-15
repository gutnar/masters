#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

#%%
def calc_gal_spin_vec(ra, dec, pos, inc):
    #pos = (pos + np.pi) % np.pi

    # Find the components of unit spin vectors in the local
    # topocentric coordinate frame.
    u = -np.sin(inc) * np.cos(pos)
    v = np.sin(inc) * np.sin(pos)
    w = np.cos(inc) # pointing away from Earth

    # Compute to alternate endpoints if the vectors were shifted
    # to the origin of the GEI frame.

    # The topocentric elevation component pointing away from Earth.
    dvec1 = np.transpose(np.array([
        -u*np.sin(ra) - v*np.sin(dec) * np.cos(ra) + w*np.cos(dec)*np.cos(ra),
        u*np.cos(ra) - v*np.sin(dec)*np.sin(ra) + w*np.cos(dec)*np.sin(ra),
        v*np.cos(dec) + w*np.sin(dec)
    ]))

    # The topocentric elevation component pointing towards Earth
    dvec2 = np.transpose(np.array([
        -u*np.sin(ra) - v*np.sin(dec)*np.cos(ra) - w*np.cos(dec)*np.cos(ra),
        u*np.cos(ra) - v*np.sin(dec)*np.sin(ra) - w*np.cos(dec)*np.sin(ra),
        v*np.cos(dec) - w*np.sin(dec)
    ]))

    # Safety procedure
    dvec1 = normalize(dvec1, axis=1)
    dvec2 = normalize(dvec2, axis=1)

    return dvec1, dvec2

#%%
def get_crd(ra, dec, dist):
    return np.array([
        dist * np.cos(ra) * np.cos(dec),
        dist * np.sin(ra) * np.cos(dec),
        dist * np.sin(dec)
    ]).T

#%%
def get_rotated_gama_coordinates_crd(crd0, gama):
    ra0 = np.select((
        gama == 9,
        gama == 12,
        gama == 15
    ), (135.0, 180.0, 217.5))
    
    dec0 = np.select((
        gama == 9,
        gama == 12,
        gama == 15
    ), (0.5, -0.5, 0.5))

    #crd = get_crd(ra0, dec0, 1.0)
    #los = crd/np.sqrt(np.sum(crd**2)) # line of sight vector
    los = get_crd(ra0, dec0, 1.0)
    crd = get_crd(ra0, dec0 - 1.0, 100.0)
    crd2 = get_crd(ra0, dec0 + 1.0, 100.0)
    crd = crd2 - crd
    #dum = np.vdot(crd, los) # crd projektsioon los suunal
    #print(crd, los)
    dum = np.sum(crd*los, axis=1)
    #crd2 = dum * los
    crd2 = (los.T * dum).T
    ex = normalize(crd - crd2, axis=1)
    #ex = crd/np.sqrt(np.sum(crd**2)) # vaatekiirega ristuv vektor
    ey = normalize(np.array([
        los[:,1] * ex[:,2] - los[:,2] * ex[:,1],
        los[:,2] * ex[:,0] - los[:,0] * ex[:,2],
        los[:,0] * ex[:,1] - los[:,1] * ex[:,0]
    ]).T, axis=1)
    #ey = ey/np.sqrt(np.sum(ey**2))
    
	# risti vektorid on los, ex, ey
	# projekteerime koordinaadid ümber
    return np.array([
        np.sum(crd0 * ex, axis=1),
        np.sum(crd0 * ey, axis=1),
        np.sum(crd0 * los, axis=1)
    ]).T

#%%
def get_dum(ra, dec, phi, theta, gama, ex, ey, ez):
    dvec1, dvec2 = calc_gal_spin_vec(
        ra/180*np.pi, dec/180*np.pi, phi, theta
    )

    crd1 = get_rotated_gama_coordinates_crd(dvec1, gama)
    crd2 = get_rotated_gama_coordinates_crd(dvec2, gama)

    fil = np.array([
        ex, ey, ez
    ]).T

    return (
        np.abs(np.sum(crd1 * fil, axis=1)),
        np.abs(np.sum(crd2 * fil, axis=1))
    )

#%%
galaxies = pd.read_csv("data/intermediate/filament_galaxies.csv")

plt.hist(np.concatenate(get_dum(
    galaxies["ra"],
    galaxies["dec"],
    galaxies["pos"]/180*np.pi,
    np.arccos(galaxies["ba"]),
    galaxies["gama"],
    galaxies["ex"],
    galaxies["ey"],
    galaxies["ez"]
)), 100, density=True, histtype="step")

#%%
galaxies = pd.read_csv("data/intermediate/filament_galaxies_classifier.csv")

plt.ylim((0.9, 1.1))
plt.hist(np.concatenate(get_dum(
    galaxies["ra"],
    galaxies["dec"],
    galaxies["p"],
    galaxies["t"],
    galaxies["gama"],
    galaxies["ex"],
    galaxies["ey"],
    galaxies["ez"]
)), 100, density=True, histtype="step")
