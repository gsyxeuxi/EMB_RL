# Reimporting necessary libraries and re-executing the code
import numpy as np
import matplotlib.pyplot as plt

# Redefining the friction force function
def friction_force(v, Fc, Fv):
    # return Fc * np.tanh(1000*v) + Fv * v
    return Fc * np.sign(v) + Fv * v    

def friction_force_stribeck(v, Fc, Fs, Fv, vs, delta_s):
    return (Fc + (Fs - Fc) * np.exp(-np.abs(v / vs) ** delta_s)) * np.sign(v) + Fv * v

def jacobian22(v, Fc, Fv, theta, J):
    return - (theta *  Fc * (1-(np.tanh(theta * v))**2) + Fv) / J

def jacobian22_stribeck(v, Fc, Fs, Fv, theta, J):
    return - (theta * (Fc + (Fs - Fc) * np.exp(-np.abs(v / vs) ** delta_s))* (1-(np.tanh(theta * v))**2) + (Fs - Fc) * (-2 * v / vs**2 * np.exp(-np.abs(v / vs) ** delta_s)) + Fv) / J

# Parameters (these are typical example values; adjust as needed)

Fc = 10.37e-3  # Coulomb friction
Fs = 1.2 * Fc  # Static friction
Fv = 2.16e-5  # Viscous friction coefficient
vs = 30  # Stribeck velocity
delta_s = 2.0  # Shape parameter
J = 4.624e-06
theta = 100
# Velocity range for plotting
v_values = np.linspace(-200, 200, 4000)

# Compute the friction force for the velocity range
F_values = friction_force_stribeck(v_values, Fc, Fs, Fv, vs, delta_s)
# j22 = jacobian22_stribeck(v_values, Fc, Fs, Fv, theta, J)
# print(theta *  Fc * (1-(np.tanh(theta * 0.05))**2))
print(jacobian22_stribeck(0.05, Fc, Fs, Fv, theta, J))
# Plot the friction force vs velocity
plt.figure(figsize=(8, 6))
plt.plot(v_values, F_values, label=r'$T_f$', color='b')
# plt.plot(v_values, j22, label=r'$F(v)$', color='r')
# plt.title(r'Friction Torque $F(v)$ vs Velocity $v$', fontsize=16)
plt.xlabel('Motor velocity (rad/s)', fontsize=14)
plt.ylabel('Friction Torque (N$\cdot$m)', fontsize=14)
plt.grid(True)
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.legend()

plt.savefig('Stribeck friction model.svg', bbox_inches='tight')
plt.show()
