#%%
from sympy import Symbol, cos, sin, atan, sqrt, lambdify, pi, simplify
from sympy.matrices import Matrix
from sympy.abc import xi, zeta, phi, theta, psi, p, A, B, C

#%%
M = Matrix((
    (1, 0, 0),
    (0, 1/zeta**2, 0),
    (0, 0, 1/xi**2)
))

R = Matrix((
    (-sin(phi), -cos(phi)*cos(theta), cos(phi)*sin(theta)),
    (cos(phi), -sin(phi)*cos(theta), sin(phi)*sin(theta)),
    (0, sin(theta), cos(theta))
))

M_prime = R.T * M * R
J = M_prime[0:2, 0:2]
L_prime = M_prime[0:2, 2]
L = M_prime[2, 0:2]
K = M_prime[2, 2]
E = J - L_prime * 1/K * L

#%%
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
q = sqrt(E_eigenvalues[0] / E_eigenvalues[1])
get_q = lambdify((xi, zeta, theta, phi), q)

#%%
psi = atan(E_eigenvects[0][1] / E_eigenvects[0][0])
get_psi = lambdify((xi, zeta, theta, phi), psi)

#%%
P = Matrix((
    (cos(p - psi), -sin(p - psi), 0),
    (sin(p - psi), cos(p - psi), 0),
    (0, 0, 1)
))

z_prime_vec = P.T * R.T * Matrix(3, 1, (0, 0, 1))
get_z_prime_vec = lambdify((xi, zeta, theta, phi, p), z_prime_vec)
