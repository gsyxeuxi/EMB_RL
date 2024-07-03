import numpy as np
import time
from scipy.integrate import odeint

# Define parameters
J = 4.624e-06  # Moment of inertia
k_m = 21.7e-03  # Motor constant
gamma = 1.889e-05  # Proportional constant
k_1 = 23.04  # Elasticity constant
f_c = 10.37e-3  # Coulomb friction coefficient
epsilon = 1000  # Excitation function coefficient
f_v = 2.16e-5  # Viscous friction coefficient

def f(x, u, theta):
    """
    Define the system dynamics of lumped parameter EMB model
    dx/dt = f(x, u, theta)
    state variable:
        x1: motor position
        x2: motor velocity
    input:
        u: motor current
    parameter:
        theta: km, k1, fc, fv, J, gamma, epsilon
    """
    x1, x2 = x
    km, k1, fc, fv, J, gamma, epsilon = theta
    dx1 = x2
    dx2 = (km / J) * u - (gamma * k1 / J) * x1 - (1 / J) * (fc * np.tanh(epsilon * x2) + fv * x2)
    return np.array([dx1, dx2])

def h(x):
    """
    Define the output equation
    y = h(x)
    output:
        y: motor position
    """
    return x[0]

def jacobian_f(x, u, theta):
    """
    Define the Jacobian matrix of function f, J_f
    dx/dt = f(x, u, theta)
    output: J_f
    """
    x1, x2 = x
    km, k1, fc, fv, J, gamma, epsilon = theta
    df1_dx = [0, 1]
    df2_dx = [-gamma * k1 / J, -1 / J * (fc * epsilon * (1 / np.cosh(epsilon * x2))**2 + fv)]
    return np.array([df1_dx, df2_dx])

def jacobian_h(x):
    """
    Define the Jacobian matrix of function h, J_h
    y = h(x)
    output: J_h
    """
    x1, x2 = x
    dh_dx1 = 1
    dh_dx2 = 0
    return np.array([dh_dx1, dh_dx2])

def df_dtheta(x, u, theta):
    """
    Define the matrix of df_dtheta with each parameter
    dx/dt = f(x, u, theta)
    output: df/dtheta
    """
    x1, x2 = x
    km, k1, fc, fv, J, gamma, epsilon = theta
    df1_dtheta = 0
    df_dkm = np.array([0, 1 / J * u]).reshape(2,1)
    df_dk1 = np.array([0, -gamma * x1 / J]).reshape(2,1)
    df_dfc = np.array([0, -np.tanh(epsilon * x2) / J]).reshape(2,1)
    df_dfv = np.array([0, -x2 / J]).reshape(2,1)
    # df_dkm = [0, 1 / J * u]
    # df_dk1 = [0, -gamma * x1 / J]
    # df_dfc = [0, -np.tanh(epsilon * x2) / J]
    # df_dfv = [0, -x2 / J]
    return np.array([df_dkm, df_dk1, df_dfc, df_dfv])

def sensitivity_x(J_f, df_dtheta, chi):
    """
    Define the sensitivity dx/dtheta with recursive algorithm
    chi(k+1) = J_x * chi(k) + df_dtheta
    output: chi(k+1)
    """
    for i in range(len(df_dtheta)):
        chi[i] = np.dot(J_f, chi[i]) + df_dtheta[i]
    return chi

def sensitivity_y(chi, J_h):
    """
    Define the sensitivity dy/dtheta
    dh_dtheta(k) = J_h * chi(k)
    output: dh_dtheta(k)
    """
    dh_dtheta = np.zeros((1, len(chi)))
    for i in range(len(chi)):
        dh_dtheta[0][i] = np.dot(J_h, chi[i]).item()
    return dh_dtheta

# Initial state
x0 = np.array([0.0, 0.0])

chi = np.zeros((4,2,1))

# Example input
u = 1.0

# Example parameters
theta = (k_m, k_1, f_c, f_v, J, gamma, epsilon)

# Compute the state derivatives
x_dot = f(x0, u, theta)
print("State derivatives:", x_dot)

# Compute the output
y = h(x0)
print("Output:", y)

# Compute the Jacobian matrix
# J_f = jacobian_f(x0, u, theta)
# J_h = jacobian_h(x0)
# df_theta = df_dtheta(x0, u, theta)
# chi = sensitivity_x(J_f, df_theta, chi)
# dh_theta = sensitivity_y(chi, J_h)

T = time.time()
det_T = 0.01
x = x0
for k in range(10):
    # u = k * det_T
    u = 0.01
    dx = f(x, u, theta)
    # print(dx)
    x = x + det_T * dx

    J_f = jacobian_f(x, u, theta)
    J_h = jacobian_h(x)
    df_theta = df_dtheta(x, u, theta)
    chi = sensitivity_x(J_f, df_theta, chi)
    dh_theta = sensitivity_y(chi, J_h)
    print(dh_theta)

