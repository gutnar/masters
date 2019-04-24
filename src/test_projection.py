#%%
import numpy as np
import matplotlib.pyplot as plt
from sympy import pi, trigsimp, Symbol, symbols, sqrt, solve, Eq, cos, sin, tan, lambdify, Abs, atan, re
from sympy.matrices import Matrix
from sympy.vector import Vector
from sympy import init_printing
from sympy.abc import xi, zeta, phi, theta, psi, omega

from lib.projection import *
import analytical

#init_printing()

#%%
blender_phi = np.array((0, 15, 30, 45, 60, 75, 90)) / 180 * np.pi
#blender_q = (0.9, 1.73/2, 1.57/2, 1.28/2, 0.92/2, 0.5/2, 0.1) # (0.1, 0.9, np.pi/2)
blender_q = (1.41/1.79, 1.36/1.81, 1.2/1.84, 0.98/1.87, 0.7/1.88, 0.4/1.89, 0.2/1.9) # (0.1, 0.9, np.pi/4)
blender_omega = np.array((0, -25, -45, -60, -70, -80, -90)) / 180 * np.pi + np.pi/2 # (0.1, 0.9, np.pi/4)

plt.figure(1)
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
        np.pi/4,
        np.linspace(0, 2*np.pi, 100),
    ),
    label="Binney"
)

plt.plot(blender_phi, blender_q, "o", label="Blender")
plt.gca().legend()

plt.figure(2)
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = \pi / 4$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_omega(
        0.1,
        0.9,
        np.pi/4,
        np.linspace(0, 2*np.pi, 100)
    ),
    label=r"$\omega$"
)

plt.plot(blender_phi, blender_omega, "o", label="Blender")
plt.gca().legend()

plt.figure(3)
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = \pi / 2$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_q(
        0.1,
        0.9,
        np.pi/2,
        np.linspace(0, 2*np.pi, 100)
    ),
    label="q"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        0.9,
        np.pi/2,
        np.linspace(0, 2*np.pi, 100),
    ),
    "o",
    markersize=3,
    label="Binney"
)

plt.gca().legend()

#%%
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = \pi / 3$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_q(
        0.1,
        0.9,
        np.pi/3,
        np.linspace(0, 2*np.pi, 100)
    ),
    label="q"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        0.9,
        np.pi/3,
        np.linspace(0, 2*np.pi, 100),
    ),
    label="Binney"
)

plt.gca().legend()

#%%
plt.title(r"$\xi = 0.1, \zeta=0.9, \theta = 0$")

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_q(
        0.1,
        0.9,
        0,
        np.linspace(0, 2*np.pi, 100)
    ),
    label="q"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        0.9,
        0,
        np.linspace(0, 2*np.pi, 100),
    ),
    label="Binney"
)

plt.gca().legend()

#%%
get_x_prime_vec = lambdify((theta, phi, psi), P.T * R.T * Matrix(3, 1, (1, 0, 0)))

def get_spin_vec(xi_value, zeta_value, theta_value, phi_value, p_value):
    omega_value = get_omega(xi_value, zeta_value, theta_value, phi_value)
    
    return get_x_prime_vec(
        theta_value,
        phi_value,
        p_value - (omega_value + np.pi/2)
    ).T

i = 0.2
p = 0.5

u = -np.sin(i) * np.cos(p)
v = np.sin(i) * np.sin(p)
w = np.cos(i)

get_spin_vec(
    0.2,
    1,
    np.pi/2,
    i,
    p
), np.array([u, v, w])

#%%
P.T * R.T * Matrix(3, 1, (1, 0, 0))

#%%
from sympy import pi

(P.T * R.T * Matrix(3, 1, (1, 0, 0))).subs({
    theta: pi/2
})
