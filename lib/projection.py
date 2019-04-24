#%%
from sympy import Symbol, cos, sin, atan, sqrt, lambdify, pi
from sympy.matrices import Matrix
from sympy.abc import xi, zeta, phi, theta, psi, omega

#%%
#xi = 0.1
#zeta = 0.9
#theta = pi
#phi = pi/4

#%%
Q = Matrix((
    (1/xi**2, 0, 0, 0),
    (0, 1/zeta**2, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, -1)
))

P = Matrix((
    (cos(psi), -sin(psi), 0),
    (sin(psi), cos(psi), 0),
    (0, 0, 1)
))

R = Matrix((
    (-sin(phi), -cos(phi)*cos(theta), cos(phi)*sin(theta), 0),
    (cos(phi), -sin(phi)*cos(theta), sin(phi)*sin(theta), 0),
    (0, sin(theta), cos(theta), 0),
    (0, 0, 0, 1)
))

R2 = Matrix((
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 0, 1)
)) * R.T

#A_prime = R.T * A * R
#J = A_prime[0:2, 0:2]
#L_prime = A_prime[0:2, 2]
#L = A_prime[2, 0:2]
#K = A_prime[2, 2]
#E = J - L_prime * 1/K * L

R2

#%%
E = (R2 * Q.inv() * R2.T)

E

#%%
P, D = E.diagonalize()

P, D

#%%
P * D * P.inv()

#%%
E_eigen = E.eigenvects()
E_eigen

#%%
#A = Symbol("A")
#B = Symbol("B")
#C = Symbol("C")

#Q = Matrix((
#    (A, B/2),
#    (B/2, C)
#))

#Q_eigen = Q.eigenvects()

#%%
q_expression = (
    (E_eigen[1][0]) / 
    (E_eigen[2][0])
)

omega_expression = atan(E_eigen[1][2][0][1] / E_eigen[2][2][0][0])

q_expression, omega_expression

#%%
get_q = lambdify((xi, zeta, theta, phi), q_expression)
get_omega = lambdify((xi, zeta, theta, phi), omega_expression)

#%%
#(R.T * Matrix(3, 1, (1, 0, 0)))

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
#plt.ylim((0, 1))

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

plt.title(r"$\theta = \pi / 2$")
plt.legend()