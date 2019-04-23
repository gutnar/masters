#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import trigsimp, Symbol, symbols, sqrt, solve, Eq, cos, sin, tan, lambdify, Abs, atan, re
from sympy.matrices import Matrix
from sympy.vector import Vector
from sympy import init_printing
from sympy.abc import xi, zeta, phi, theta, psi

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

Q_eigen = Q.eigenvects()

q_expression = (
    (Q_eigen[0][0]) / 
    (Q_eigen[1][0])
).subs({
    A: E[0, 0],
    B: E[1, 0],
    C: E[1, 1]
})

psi_expression = atan(Q_eigen[0][2][0][1] / Q_eigen[0][2][0][0]).subs({
    A: E[0, 0],
    B: E[1, 0],
    C: E[1, 1]
})

#%%
get_q = lambdify((xi, zeta, theta, phi), q_expression)

plt.ylim((0, 1))

plt.plot(
    np.linspace(0, 2*np.pi, 100),
    get_q(
        0.1,
        0.9,
        0,
        np.linspace(0, 2*np.pi, 100),
    )
)

get_q(0.1, 0.9, np.pi/2, np.pi/2)

#%%
get_psi = lambdify((xi, zeta, theta, phi), psi_expression)

x = np.linspace(0, 2*np.pi, 1000)
y = get_psi(
    0.1,
    0.9,
    np.pi/2,
    np.linspace(0, 2*np.pi, 1000)
)

plt.plot(x, y)

#%%
q = Symbol("q")

#sol = solve((
#    Eq(q_expression, q),
#    Eq(psi_expression, psi)
#), (theta, phi))

#%%
R * Matrix(3, 1, (1, 0, 0))

#%%
import analytical

plt.figure(1)
plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_q(
        0.1,
        0.9,
        0,
        np.linspace(0, 2*np.pi, 100),
    )
)

plt.figure(2)
plt.plot(
    np.linspace(0, 2*np.pi, 100),
    analytical.get_psi(
        0.1,
        0.9,
        0,
        np.linspace(0, 2*np.pi, 100)
    )
)

#%%
solve(Eq(q_expression, q), theta)
