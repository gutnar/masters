#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, sqrt, solve, Eq, cos, sin, tan, lambdify, Abs, atan, re
from sklearn.preprocessing import normalize

x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)
p = Symbol("p", positive=True)
t = Symbol("t", positive=True)

A = 1/x**2*cos(t)**2*sin(p)**2 + 1/x**2*1/z**2*cos(t)**2*cos(p)**2 + 1/z**2*sin(t)**2
B = 1/x**2*cos(t)*sin(2*p) - 1/x**2*1/z**2*cos(t)*sin(2*p)
C = 1/x**2*cos(p)**2 + 1/x**2*1/z**2*sin(p)**2
D = sqrt((A - C)**2 + B**2)
Q = (1 - q**2) / (1 + q**2)
E = Q**2 * (A+C)**2 - (A - C)**2 + B**2

#solution = solve(Eq(E, 0), cos(t))

#%%
q_expression = sqrt((A + C - D) / (A + C + D))
get_q = lambdify([x, z, p, t], q_expression, "numpy")

#%%
psi_expression = 1/2*atan(B/(A - C))
get_psi = lambdify([x, z, p, t], psi_expression, "numpy")

#%%
cos_t_expression = sqrt(q**2*x**2*z**2*sin(p)**2/2 - q**2*x**2*z**2/2 - q**2*x**2*sin(p)**2/2 + q**2*z**2/2 + x**4 - x**2*z**2*sin(p)**2 + x**2*sin(p)**2 - x**2 + z**4*sin(p)**4 - z**4*sin(p)**2 - 2*z**2*sin(p)**4 + 2*z**2*sin(p)**2 - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/2 + sin(p)**4 - sin(p)**2 + x**2*z**2*sin(p)**2/(2*q**2) - x**2*z**2/(2*q**2) - x**2*sin(p)**2/(2*q**2) + z**2/(2*q**2) - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/(2*q**2))/Abs(x**2 - z**2*sin(p)**2 + sin(p)**2 - 1)
get_cos_t = lambdify([q, x, z, p], cos_t_expression, "numpy")

#%%
p_root_expression = re(atan(sqrt((x**2 - q**2*z**2)/(q**2 - x**2))))
get_p_root = lambdify([q, x, z], p_root_expression, "numpy")

def get_p_domain(q_values, x_values, z_values):
    roots = get_p_root(q_values, x_values, z_values)
    
    if len(roots.shape):
        roots[np.isnan(roots)] = 0
    elif np.isnan(roots):
        roots = 0

    return roots, np.pi - roots

#%%
if __name__ == "__main__":
    q_values = np.array([0.21, 0.3, 0.5, 0.895])
    x_value = 0.2
    z_value = 0.9

    for q_value in q_values:
        p1, p2 = get_p_domain(q_value, x_value, z_value)
        p_values = np.linspace(p1, p2, 1000, False)

        plt.plot(
            p_values, get_cos_t(q_value, x_value, z_value, p_values)
        )

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
