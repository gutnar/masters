#%%
from sympy import Symbol, cos, sin, atan, sqrt, lambdify, pi, simplify
from sympy.matrices import Matrix
from sympy.abc import xi, zeta, phi, theta, psi, omega

#%%
A = Matrix((
    (1/xi**2, 0, 0),
    (0, 1/zeta**2, 0),
    (0, 0, 1)
))

P = Matrix((
    (cos(psi), -sin(psi), 0),
    (sin(psi), cos(psi), 0),
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
E = J - L_prime * 1/K * L

#%%
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")

Q = Matrix((
    (A, B/2),
    (B/2, C)
))

Q_eigen = Q.eigenvects()

subs = {
    A: E[0, 0],
    B: E[0, 1] + E[1, 0],
    C: E[1, 1]
}

E_eigenvalues = (
    Q_eigen[0][0].subs(subs),
    Q_eigen[1][0].subs(subs)
)

E_eigenvects = (
    Q_eigen[0][2][0].subs(subs),
    Q_eigen[1][2][0].subs(subs)
)

#%%
q_expression = sqrt(E_eigenvalues[0] / E_eigenvalues[1])
get_q = lambdify((xi, zeta, theta, phi), q_expression)

#%%
omega_expression = atan(E_eigenvects[1][1] / E_eigenvects[1][0])
get_omega = lambdify((xi, zeta, theta, phi), omega_expression)
