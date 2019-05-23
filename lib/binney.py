#%%
from sympy import sqrt, cos, sin, lambdify, atan, sign
from sympy.abc import xi, zeta, theta, phi

A = 1/xi**2*cos(theta)**2*sin(phi)**2 + 1/xi**2*1/zeta**2*cos(theta)**2*cos(phi)**2 + 1/zeta**2*sin(theta)**2
B = 1/xi**2*cos(theta)*sin(2*phi) - 1/xi**2*1/zeta**2*cos(theta)*sin(2*phi)
C = 1/xi**2*cos(phi)**2 + 1/xi**2*1/zeta**2*sin(phi)**2
D = sqrt((A - C)**2 + B**2)

#%%
q = sqrt((A + C - D) / (A + C + D))
get_q = lambdify([xi, zeta, theta, phi], q, "numpy")

#%%
psi = 1/2*atan(B/(A - C))
get_psi = lambdify([xi, zeta, theta, phi], psi, "numpy")
