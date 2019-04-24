#%%
from sympy import Symbol, cos, sin, atan, sqrt, lambdify, pi, solve, expand, collect, simplify
from sympy.matrices import Matrix
from sympy.abc import x, y, z, xi, zeta, phi, theta, psi, omega

#%%
Q = Matrix((
    (1/xi**2, 0, 0),
    (0, 1/zeta**2, 0),
    (0, 0, 1)
))

s = Matrix(3, 1, (
    sin(theta)*cos(phi),
    sin(theta)*sin(phi),
    cos(theta)
))

vec = Matrix(3, 1, (
    x,
    y,
    solve(Matrix(1, 3, (x, y, z)) * s, z)[z]
))

f = (vec.T * Q * s)**2 / (s.T * Q * s) - (vec.T * Q * vec) # = 1
f[0]

#%%
collected = collect(expand(f[0]), (x**2, x*y, y**2))

A = simplify(collected.coeff(x, 2))
B = simplify(collected.coeff(x, 1) / y)
C = simplify(collected.coeff(y, 2))

A, B, C

#%%
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")

E = Matrix((
    (A, B/2),
    (B/2, C)
))

P, D = E.diagonalize()

components = {
    A: simplify(collected.coeff(x, 2)),
    B: simplify(collected.coeff(x, 1) / y),
    C: simplify(collected.coeff(y, 2))
}

q = sqrt(simplify(D[1, 1] / D[0, 0])).subs(components)

#%%
get_q = lambdify((xi, zeta, theta, phi), q)

#%%
import matplotlib.pyplot as plt
import numpy as np
import analytical

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
