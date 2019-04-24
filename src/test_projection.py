#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import pi, trigsimp, Symbol, symbols, sqrt, solve, Eq, cos, sin, tan, lambdify, Abs, atan, re
from sympy.matrices import Matrix
from sympy.vector import Vector
from sympy import init_printing
from sympy.abc import xi, zeta, phi, theta, psi, omega

from lib.projection import *
import analytical

init_printing()

#%%
plt.ylim((0, 1))

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_q(
        0.1,
        1,
        np.pi/2,
        np.linspace(0, 2*np.pi, 100)
    )
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    np.maximum(0.1, np.abs(np.cos(np.linspace(0, 2*np.pi, 100))))
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        1,
        0,
        np.linspace(0, 2*np.pi, 100),
    ),
    label="Binney"
)

#plt.plot(
#    np.linspace(0, 2*np.pi, 100),
#    np.sqrt(np.cos(np.linspace(0, 2*np.pi, 100))**2 * (1 - 0.1**2) + 0.1**2)
#)

plt.title(r"$\theta = \pi / 2$")
plt.legend()

#%%
x = np.linspace(0, 2*np.pi, 1000)
y = get_omega(
    0.1,
    0.9,
    0,
    np.linspace(0, 2*np.pi, 1000)
)

plt.plot(x, y)

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

#%%
plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        0.9,
        np.pi/2,
        np.linspace(0, 2*np.pi, 100),
    ),
    label="q"
)

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_psi(
        0.1,
        0.9,
        np.pi/5,
        np.linspace(0, 2*np.pi, 100),
    ),
    label="psi"
)

plt.legend()

#%%

