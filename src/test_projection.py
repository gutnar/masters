#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import trigsimp, Symbol, symbols, sqrt, solve, Eq, cos, sin, tan, lambdify, Abs, atan, re
from sympy.matrices import Matrix
from sympy.vector import Vector
from sympy import init_printing
from sympy.abc import xi, zeta, phi, theta

init_printing()

#%%
#x_prime = symbols("x'")
#y_prime = symbols("y'")
#z_prime = symbols("z'")
#xy_vec = Matrix(2, 1, (x_prime, y_prime))

A = Matrix((
    (1/xi, 0, 0),
    (0, 1/zeta, 0),
    (0, 0, 1)
))

R = Matrix((
    (-sin(phi), -cos(phi)*cos(theta), cos(phi)*sin(theta)),
    (cos(phi), -sin(phi)*cos(theta), sin(phi)*sin(theta)),
    (0, sin(theta), cos(theta))
))

A_prime = R.T * A * R
J = A_prime[0:2, 0:2]
L_prime = A_prime[0:2, 2]
L = A_prime[2, 0:2]
K = A_prime[2, 2]

E = J - Matrix(((1, 0), (0, 1))) * L_prime * 1/K * L

#Eq((xy_vec.T * E * xy_vec)[0], 1)

#%%
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")

Q = Matrix((
    (A, B),
    (B, C)
))

P, D = Q.diagonalize()

P = P.subs({
    A: E[0, 0],
    B: E[1, 0],
    C: E[1, 1]
})

D = D.subs({
    A: E[0, 0],
    B: E[1, 0],
    C: E[1, 1]
})

q = D[0, 0] / D[1, 1]

Q_eigen = Q.eigenvects()

#%%
q = (
    (Q_eigen[0][0]) / 
    (Q_eigen[1][0])
).subs({
    A: E[0, 0],
    B: E[1, 0],
    C: E[1, 1]
})

get_q = lambdify((xi, zeta, theta, phi), q)

plt.ylim((0, 1))

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_q(
        0.1,
        0.9,
        np.pi/2,
        np.linspace(0, 2*np.pi, 100),
    )
)

get_q(0.1, 0.9, np.pi/2, np.pi/2)

#%%
import analytical

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        0.9,
        0,
        np.linspace(0, 2*np.pi, 100),
    )
)

#%%

