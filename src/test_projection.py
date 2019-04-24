
#%%
import numpy as np
import matplotlib.pyplot as plt
from sympy import init_printing

from lib.projection import *
import analytical

#init_printing()

#%%
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = \pi / 4$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_q(
        0.1,
        0.9,
        np.pi/4,
        np.linspace(0, 2*np.pi, 100)
    ),
    label="q"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        0.9,
        np.linspace(0, 2*np.pi, 100),
        np.pi/4,
    ),
    "o",
    markersize=2,
    label="Binney"
)

plt.legend()

#%%
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = 0$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_psi(
        0.1,
        0.9,
        0,
        np.linspace(0, 2*np.pi, 100)
    ),
    label=r"$\psi$"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_psi(
        0.1,
        0.9,
        np.linspace(0, 2*np.pi, 100),
        0,
    ),
    "o",
    markersize=2,
    label="Binney"
)

plt.legend()

#%%
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = \pi / 4$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_psi(
        0.1,
        0.9,
        np.pi/4,
        np.linspace(0, 2*np.pi, 100)
    ),
    label=r"$\psi$"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_psi(
        0.1,
        0.9,
        np.linspace(0, 2*np.pi, 100),
        np.pi/4,
    ),
    "o",
    markersize=2,
    label="Binney"
)

plt.legend()

#%%
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = \pi / 2$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_psi(
        0.5,
        0.9,
        np.pi/5*8,
        np.linspace(0, 2*np.pi, 100)
    ),
    label=r"$\psi$"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_psi(
        0.5,
        0.9,
        np.linspace(0, 2*np.pi, 100),
        np.pi/5*8,
    ),
    "o",
    markersize=2,
    label="Binney"
)

plt.legend()

#%%
i = 0.6
p = 0.8

u = -np.sin(i) * np.cos(p)
v = np.sin(i) * np.sin(p)
w = np.cos(i)

get_z_prime_vec(
    0.2,
    1,
    i,
    0,
    p
).T[0], np.array([u, v, w])

#%%
z_prime_vec.subs({
    xi: 0.1,
    zeta: 0.9,
    theta: 0.123,
    phi: 0.666,
    p: 0
})

#%%
np.cos(i)

#%%
R.T * Matrix(3, 1, (1, 0, 0))

#%%
simplify(z_prime_vec.subs({
    zeta: 1,
    psi: 0,
    phi: 0
}))
