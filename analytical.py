#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import init_printing, Symbol, symbols, sqrt, solve, solveset, S, Eq, cos, sin, tan, lambdify, simplify, trigsimp, Abs, Poly, acos
from sympy.solvers.solveset import nonlinsolve
from sympy.solvers.solvers import nsolve
from scipy.optimize import minimize

#init_printing()

#%%
x = Symbol("x", positive=True)
z = Symbol("z", positive=True)
q = Symbol("q", positive=True)
p = Symbol("p", positive=True)
t = Symbol("t", positive=True)

A = 1/x**2*cos(t)**2*sin(p)**2 + 1/x**2*1/z**2*cos(t)**2*cos(p)**2 + 1/z**2*sin(t)**2
B = 1/x**2*cos(t)*sin(2*p) - 1/x**2*1/z**2*cos(t)*sin(2*p)
C = 1/x**2*cos(p)**2 + 1/x**2*1/z**2*sin(p)**2
Q = (1 - q**2) / (1 + q**2)
E = Q**2 * (A+C)**2 - (A - C)**2 + B**2

#%%
solution = solve(Eq(E, 0), cos(t))

#%%
solution

#%%
solution = solveset(
    E,
    cos(t),
    domain=S.Reals
)

#%%
Intersection(Reals, {
    -sqrt(-sqrt(16*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2*(4*q**4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*q**4*x**2*sin(p)**2*sin(t)**2 - 4*q**2*x**4*sin(t)**4 - 4*q**2*z**4*cos(p)**4 - q**2*z**2*(-cos(4*p) + 1) - 4*q**2*sin(p)**4 + 4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*x**2*sin(p)**2*sin(t)**2) + (q**4*z**4*(-cos(4*p) + 1) + 2*q**4*z**4*sin(2*p)**2 + 8*q**4*z**2*sin(p)**4 - 4*q**4*z**2*sin(2*p)**2 + 8*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1) + 2*q**4*sin(2*p)**2 - 16*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 16*q**2*x**2*sin(t)**2*cos(p)**2 + 4*q**2*z**4*sin(2*p)**2 - 8*q**2*z**2*sin(2*p)**2 + 4*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1) + 2*z**4*sin(2*p)**2 + 8*z**2*sin(p)**4 - 4*z**2*sin(2*p)**2 + 8*z**2*cos(p)**4 + 2*sin(2*p)**2 - cos(4*p) + 1)**2/4)/(8*q**2*(z**4*sin(p)**4 + 2*z**2*sin(p)**2*cos(p)**2 + cos(p)**4)) + (q**4*z**4*(-cos(4*p) + 1)/2 + q**4*z**4*sin(2*p)**2 + 4*q**4*z**2*sin(p)**4 - 2*q**4*z**2*sin(2*p)**2 + 4*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1)/2 + q**4*sin(2*p)**2 - 8*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 8*q**2*x**2*sin(t)**2*cos(p)**2 + 2*q**2*z**4*sin(2*p)**2 - 4*q**2*z**2*sin(2*p)**2 + 2*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1)/2 + z**4*sin(2*p)**2 + 4*z**2*sin(p)**4 - 2*z**2*sin(2*p)**2 + 4*z**2*cos(p)**4 + sin(2*p)**2 - cos(4*p)/2 + 1/2)/(8*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2)), sqrt(-sqrt(16*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2*(4*q**4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*q**4*x**2*sin(p)**2*sin(t)**2 - 4*q**2*x**4*sin(t)**4 - 4*q**2*z**4*cos(p)**4 - q**2*z**2*(-cos(4*p) + 1) - 4*q**2*sin(p)**4 + 4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*x**2*sin(p)**2*sin(t)**2) + (q**4*z**4*(-cos(4*p) + 1) + 2*q**4*z**4*sin(2*p)**2 + 8*q**4*z**2*sin(p)**4 - 4*q**4*z**2*sin(2*p)**2 + 8*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1) + 2*q**4*sin(2*p)**2 - 16*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 16*q**2*x**2*sin(t)**2*cos(p)**2 + 4*q**2*z**4*sin(2*p)**2 - 8*q**2*z**2*sin(2*p)**2 + 4*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1) + 2*z**4*sin(2*p)**2 + 8*z**2*sin(p)**4 - 4*z**2*sin(2*p)**2 + 8*z**2*cos(p)**4 + 2*sin(2*p)**2 - cos(4*p) + 1)**2/4)/(8*q**2*(z**4*sin(p)**4 + 2*z**2*sin(p)**2*cos(p)**2 + cos(p)**4)) + (q**4*z**4*(-cos(4*p) + 1)/2 + q**4*z**4*sin(2*p)**2 + 4*q**4*z**2*sin(p)**4 - 2*q**4*z**2*sin(2*p)**2 + 4*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1)/2 + q**4*sin(2*p)**2 - 8*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 8*q**2*x**2*sin(t)**2*cos(p)**2 + 2*q**2*z**4*sin(2*p)**2 - 4*q**2*z**2*sin(2*p)**2 + 2*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1)/2 + z**4*sin(2*p)**2 + 4*z**2*sin(p)**4 - 2*z**2*sin(2*p)**2 + 4*z**2*cos(p)**4 + sin(2*p)**2 - cos(4*p)/2 + 1/2)/(8*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2)), -sqrt(sqrt(16*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2*(4*q**4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*q**4*x**2*sin(p)**2*sin(t)**2 - 4*q**2*x**4*sin(t)**4 - 4*q**2*z**4*cos(p)**4 - q**2*z**2*(-cos(4*p) + 1) - 4*q**2*sin(p)**4 + 4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*x**2*sin(p)**2*sin(t)**2) + (q**4*z**4*(-cos(4*p) + 1) + 2*q**4*z**4*sin(2*p)**2 + 8*q**4*z**2*sin(p)**4 - 4*q**4*z**2*sin(2*p)**2 + 8*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1) + 2*q**4*sin(2*p)**2 - 16*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 16*q**2*x**2*sin(t)**2*cos(p)**2 + 4*q**2*z**4*sin(2*p)**2 - 8*q**2*z**2*sin(2*p)**2 + 4*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1) + 2*z**4*sin(2*p)**2 + 8*z**2*sin(p)**4 - 4*z**2*sin(2*p)**2 + 8*z**2*cos(p)**4 + 2*sin(2*p)**2 - cos(4*p) + 1)**2/4)/(8*q**2*(z**4*sin(p)**4 + 2*z**2*sin(p)**2*cos(p)**2 + cos(p)**4)) + (q**4*z**4*(-cos(4*p) + 1)/2 + q**4*z**4*sin(2*p)**2 + 4*q**4*z**2*sin(p)**4 - 2*q**4*z**2*sin(2*p)**2 + 4*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1)/2 + q**4*sin(2*p)**2 - 8*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 8*q**2*x**2*sin(t)**2*cos(p)**2 + 2*q**2*z**4*sin(2*p)**2 - 4*q**2*z**2*sin(2*p)**2 + 2*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1)/2 + z**4*sin(2*p)**2 + 4*z**2*sin(p)**4 - 2*z**2*sin(2*p)**2 + 4*z**2*cos(p)**4 + sin(2*p)**2 - cos(4*p)/2 + 1/2)/(8*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2)), sqrt(sqrt(16*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2*(4*q**4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*q**4*x**2*sin(p)**2*sin(t)**2 - 4*q**2*x**4*sin(t)**4 - 4*q**2*z**4*cos(p)**4 - q**2*z**2*(-cos(4*p) + 1) - 4*q**2*sin(p)**4 + 4*x**2*z**2*sin(t)**2*cos(p)**2 + 4*x**2*sin(p)**2*sin(t)**2) + (q**4*z**4*(-cos(4*p) + 1) + 2*q**4*z**4*sin(2*p)**2 + 8*q**4*z**2*sin(p)**4 - 4*q**4*z**2*sin(2*p)**2 + 8*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1) + 2*q**4*sin(2*p)**2 - 16*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 16*q**2*x**2*sin(t)**2*cos(p)**2 + 4*q**2*z**4*sin(2*p)**2 - 8*q**2*z**2*sin(2*p)**2 + 4*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1) + 2*z**4*sin(2*p)**2 + 8*z**2*sin(p)**4 - 4*z**2*sin(2*p)**2 + 8*z**2*cos(p)**4 + 2*sin(2*p)**2 - cos(4*p) + 1)**2/4)/(8*q**2*(z**4*sin(p)**4 + 2*z**2*sin(p)**2*cos(p)**2 + cos(p)**4)) + (q**4*z**4*(-cos(4*p) + 1)/2 + q**4*z**4*sin(2*p)**2 + 4*q**4*z**2*sin(p)**4 - 2*q**4*z**2*sin(2*p)**2 + 4*q**4*z**2*cos(p)**4 + q**4*(-cos(4*p) + 1)/2 + q**4*sin(2*p)**2 - 8*q**2*x**2*z**2*sin(p)**2*sin(t)**2 - 8*q**2*x**2*sin(t)**2*cos(p)**2 + 2*q**2*z**4*sin(2*p)**2 - 4*q**2*z**2*sin(2*p)**2 + 2*q**2*sin(2*p)**2 + z**4*(-cos(4*p) + 1)/2 + z**4*sin(2*p)**2 + 4*z**2*sin(p)**4 - 2*z**2*sin(2*p)**2 + 4*z**2*cos(p)**4 + sin(2*p)**2 - cos(4*p)/2 + 1/2)/(8*q**2*(z**2*sin(p)**2 - sin(p)**2 + 1)**2))
})


#%%
solution

#%%
solution = (
    (
        (-(q**2 + 1)*(t - 1)*(t + 1)*(t**2 - 1)*sqrt(q**4*z**4*sin(p)**4 - 2*q**4*z**4*sin(p)**2 + q**4*z**4 - 2*q**4*z**2*sin(p)**4 + 2*q**4*z**2*sin(p)**2 + q**4*sin(p)**4 + 4*q**2*t**2*z**4*sin(p)**4 - 4*q**2*t**2*z**4*sin(p)**2 - 8*q**2*t**2*z**2*sin(p)**4 + 8*q**2*t**2*z**2*sin(p)**2 + 4*q**2*t**2*sin(p)**4 - 4*q**2*t**2*sin(p)**2 - 2*q**2*z**4*sin(p)**4 + 4*q**2*z**4*sin(p)**2 - 2*q**2*z**4 + 4*q**2*z**2*sin(p)**4 - 4*q**2*z**2*sin(p)**2 - 2*q**2*sin(p)**4 + z**4*sin(p)**4 - 2*z**4*sin(p)**2 + z**4 - 2*z**2*sin(p)**4 + 2*z**2*sin(p)**2 + sin(p)**4) + (t**4 - 2*t**2 + 1)*(-q**4*z**2*cos(p)**2 - q**4*sin(p)**2 + 2*q**2*t**2*z**2*sin(p)**2 + 2*q**2*t**2*cos(p)**2 - z**2*cos(p)**2 - sin(p)**2))/(2*q**2*(t**2 - 1)*(t**4 - 2*t**2 + 1)),# \ {0}
        z**2
    ),
    (
        ((q**2 + 1)*(t - 1)*(t + 1)*(t**2 - 1)*sqrt(q**4*z**4*sin(p)**4 - 2*q**4*z**4*sin(p)**2 + q**4*z**4 - 2*q**4*z**2*sin(p)**4 + 2*q**4*z**2*sin(p)**2 + q**4*sin(p)**4 + 4*q**2*t**2*z**4*sin(p)**4 - 4*q**2*t**2*z**4*sin(p)**2 - 8*q**2*t**2*z**2*sin(p)**4 + 8*q**2*t**2*z**2*sin(p)**2 + 4*q**2*t**2*sin(p)**4 - 4*q**2*t**2*sin(p)**2 - 2*q**2*z**4*sin(p)**4 + 4*q**2*z**4*sin(p)**2 - 2*q**2*z**4 + 4*q**2*z**2*sin(p)**4 - 4*q**2*z**2*sin(p)**2 - 2*q**2*sin(p)**4 + z**4*sin(p)**4 - 2*z**4*sin(p)**2 + z**4 - 2*z**2*sin(p)**4 + 2*z**2*sin(p)**2 + sin(p)**4) + (t**4 - 2*t**2 + 1)*(-q**4*z**2*cos(p)**2 - q**4*sin(p)**2 + 2*q**2*t**2*z**2*sin(p)**2 + 2*q**2*t**2*cos(p)**2 - z**2*cos(p)**2 - sin(p)**2))/(2*q**2*(t**2 - 1)*(t**4 - 2*t**2 + 1)),# \ {0},
        z**2
    )
)

solution

#%%
x1 = sqrt(solution[0][0])
x2 = sqrt(solution[1][0])
xlambdified = lambdify([q, t, p, z], x2, "numpy")

plt.plot(
    np.linspace(0, 1, 100),
    xlambdified(
        0.2,
        np.arccos(np.random.uniform(0, 1)),
        np.random.uniform(0, 2*np.pi),
        np.linspace(0, 1, 100)
    )
)

#%%
x2


#%%
expression = sqrt(q**2*x**2*z**2*sin(p)**2/2 - q**2*x**2*z**2/2 - q**2*x**2*sin(p)**2/2 + q**2*z**2/2 + x**4 - x**2*z**2*sin(p)**2 + x**2*sin(p)**2 - x**2 + z**4*sin(p)**4 - z**4*sin(p)**2 - 2*z**2*sin(p)**4 + 2*z**2*sin(p)**2 - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/2 + sin(p)**4 - sin(p)**2 + x**2*z**2*sin(p)**2/(2*q**2) - x**2*z**2/(2*q**2) - x**2*sin(p)**2/(2*q**2) + z**2/(2*q**2) - sqrt(q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6)/(2*q**2))/Abs(x**2 - z**2*sin(p)**2 + sin(p)**2 - 1)

f = lambdify([q, x, z, p], expression, "numpy")

#%%
p_values = np.linspace(0, np.pi, 1000)

plt.plot(
    p_values,
    f(0.21, 0.2, 0.9, p_values)
)

plt.plot(
    p_values,
    f(0.3, 0.2, 0.9, p_values)
)

plt.plot(
    p_values,
    f(0.5, 0.2, 0.9, p_values)
)

plt.plot(
    p_values,
    f(0.895, 0.2, 0.9, p_values)
)

plt.plot((0, np.pi), (0, 0))

#%%
f(0.21, 0.2, 0.9, p_values)

#%%
sqrt((-Q*x**2 - Q + x**2 - 1)/(Q*z**2 - Q + z**2 - 1)).subs({
    q: 0.2,
    x: 0.19,
    z: 0.9
})


#%%
nsolve(
    (q**4*x**4*z**4*sin(p)**4 - 2*q**4*x**4*z**4*sin(p)**2 + q**4*x**4*z**4 - 2*q**4*x**4*z**2*sin(p)**4 + 2*q**4*x**4*z**2*sin(p)**2 + q**4*x**4*sin(p)**4 - 2*q**4*x**2*z**4*sin(p)**6 + 6*q**4*x**2*z**4*sin(p)**4 - 4*q**4*x**2*z**4*sin(p)**2 - 2*q**4*x**2*z**4*cos(p)**6 - 2*q**4*x**2*z**2*sin(p)**2 + 3*q**4*z**4*sin(p)**8 - 8*q**4*z**4*sin(p)**6 + 6*q**4*z**4*sin(p)**4 - 3*q**4*z**4*cos(p)**8 + 4*q**4*z**4*cos(p)**6 + 2*q**2*x**4*z**4*sin(p)**4 - 2*q**2*x**4*z**4 - 4*q**2*x**4*z**2*sin(p)**4 + 4*q**2*x**4*z**2*sin(p)**2 + 2*q**2*x**4*sin(p)**4 - 4*q**2*x**4*sin(p)**2 - 4*q**2*x**2*z**6*sin(p)**4 + 4*q**2*x**2*z**6*sin(p)**2 + 4*q**2*x**2*z**4*sin(p)**6 - 8*q**2*x**2*z**4*sin(p)**4 + 4*q**2*x**2*z**4*sin(p)**2 + 4*q**2*x**2*z**4*cos(p)**6 + 4*q**2*x**2*z**2*sin(p)**4 - 4*q**2*x**2*sin(p)**4 + 4*q**2*x**2*sin(p)**2 - 4*q**2*z**6*sin(p)**8 + 12*q**2*z**6*sin(p)**6 - 8*q**2*z**6*sin(p)**4 + 4*q**2*z**6*cos(p)**8 - 4*q**2*z**6*cos(p)**6 + 2*q**2*z**4*sin(p)**8 - 8*q**2*z**4*sin(p)**6 + 4*q**2*z**4*sin(p)**4 - 2*q**2*z**4*cos(p)**8 - 4*q**2*z**2*sin(p)**8 + 12*q**2*z**2*sin(p)**6 - 8*q**2*z**2*sin(p)**4 + 4*q**2*z**2*cos(p)**8 - 4*q**2*z**2*cos(p)**6 + x**4*z**4*sin(p)**4 - 2*x**4*z**4*sin(p)**2 + x**4*z**4 - 2*x**4*z**2*sin(p)**4 + 2*x**4*z**2*sin(p)**2 + x**4*sin(p)**4 - 2*x**2*z**4*sin(p)**6 + 6*x**2*z**4*sin(p)**4 - 4*x**2*z**4*sin(p)**2 - 2*x**2*z**4*cos(p)**6 - 2*x**2*z**2*sin(p)**2 + 3*z**4*sin(p)**8 - 8*z**4*sin(p)**6 + 6*z**4*sin(p)**4 - 3*z**4*cos(p)**8 + 4*z**4*cos(p)**6).subs({
        q: 0.2,
        x: 0.19,
        z: 0.9
    }), [sin(p)])
