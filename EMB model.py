import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define parameters
J = 4.624e-06  # Moment of inertia
k_m = 21.7e-03  # Motor constant
gamma = 1.889e-05  # Proportional constant
k_1 = 23.04  # Elasticity constant
f_c = 10.37e-3  # Coulomb friction coefficient
epsilon = 1000  # Excitation function coefficient
f_v = 2.16e-5  # Viscous friction coefficient

print((f_c + f_v * 0.5)/k_m)

# Define the system dynamics function f(x, u, theta)
def f(x, t, u, theta):
    x1, x2 = x
    km, k1, fc, fv, J, gamma, epsilon = theta
    dx1 = x2
    dx2 = (km / J) * u[int(t * 100)] - (gamma * k1 / J) * x1 - (1 / J) * (fc * np.tanh(epsilon * x2) + fv * x2)
    return [dx1, dx2]

# Define the output equation h(x)
def h(x):
    return x[0]

# Initial conditions
x0 = [0, 0]

# Time vector
t = np.linspace(0, 10, 100)  # from 0 to 10 seconds, 100 points

# Input function u(t) = random(0, 0.01) * t
random_coefficients = np.random.uniform(0, 0.0001, size=t.shape)
u = random_coefficients * t

# Parameters theta
theta = (k_m, k_1, f_c, f_v, J, gamma, epsilon)

# Solve the ODEs
x = odeint(f, x0, t, args=(u, theta))

# Extract motor position x1(t)
x1 = x[:, 0]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, x1, label='Motor Position $x_1(t)$')
plt.title('Motor Position Over Time with Random Input $u(t) = random(0, 0.01) * t$')
plt.xlabel('Time $t$ (seconds)')
plt.ylabel('Position $x_1(t)$')
plt.grid(True)
plt.legend()
plt.show()
