import numpy as np
import matplotlib.pyplot as plt
import math

#T_m_max = xxx (Nm)
#F_cl_max = 10000 (N)
#motor_speed _max = xxx (rad/s)

fc = 10.37e-3;           # Coulomb friction [Nm]
fcc = 1000;              # For tanh approximation of signum function
fv = 2.16e-5;            # Viscous friction [Nm/(rad/s)]
# fl = 2.859e-06;          # Load dependent friction [Nm/N]
# fs = 0.0149;             # Static friction [Nm]
J_m = 4.624e-06;         # Moment of inertia [kg*m^2]
km = 21.7e-03;           # Motor torque constant [km]
eff = 1;                 # Gear efficiency 
gamma = eff * 1.889e-05; # Total gear ratio [Nm/N]
RC_const = 1 / (2 * math.pi * 5e4)
T_sample = 1e-4

def clamping_force(motor_pos):
    """
    Calculate the clamping_force F_cl based on the simplyfied stiffness curve.

    Simplyfied stiffness curve:
    y = p1*x^3 + p2*x^2 + p3*x + p4
    p1 = 0.00643
    p2 = 0.1919 
    p3 = 5.934  
    p4 = 0 
    x = 0: 1: 50;

    Parameters:
    m_pos (float): The angle of the motor in rad.

    Returns:
    float: The calculated force F_cl.
    """
    k1 = 23.04
    F_cl = k1 * motor_pos
    return F_cl

# # Example usage
# motor_angle = 6  # Example motor angle
# force = clamping_force(motor_angle)
# print(f"The calculated force F_cl for motor angle {motor_angle} is {force}")

def motor_current(motor_current_tgt, motor_current_past, RC_const, T_sample):
    """
    Calculate the motor current i_m based with a low pass filter as motor current controller.

    Parameters:
    motor_current: The tested current in A.
    motor_current_past: The last time tested current in A.
    RC_const: RC constant of the low pass filter.
    T_sampel: The sample time of the system.

    Returns:
    float: The calculated force F_cl.
    """
    A = T_sample / (RC_const + T_sample)
    motor_current =  A * motor_current_tgt + (1-A) * motor_current_past
    return motor_current

# # Example usage
# motor_current_past = 0
# time_values = []
# current_values = []

# for time in np.arange(0, 0.01, T_sample):  # Simulating for 0.1 seconds
#     if time == 0:
#         motor_current_tgt = 0
#     else:
#         motor_current_tgt = 1
#     motor_torque_value, motor_current_past = motor_torque(motor_current_tgt, motor_current_past, RC_const, T_sample, km)
#     time_values.append(time)
#     current_values.append(motor_current_past)

# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(time_values, current_values, label='Motor Current')
# plt.xlabel('Time (s)')
# plt.ylabel('Motor Current (A)')
# plt.title('Motor Current vs Time')
# plt.legend()
# plt.grid(True)
# plt.show()

def friction_torque(fc, fv, fcc, motor_vel):
    """
    Calculate the friction torque T_f based with a low pass filter as motor current controller.

    Parameters:
    fc: Coulomb friction [Nm]
    fcc: For tanh approximation of signum function
    fv: Viscous friction [Nm/(rad/s)]

    Returns:
    float: The calculated fricion torque T_f.
    """
    return fc * math.tanh(fcc * motor_vel) + fv * motor_vel

T_m = km * motor_current(motor_current_tgt, motor_current_past, RC_const, T_sample)
F_cl = clamping_force(motor_pos)
T_f = friction_torque(fc, fv, fcc, motor_vel)
motor_acc = 1/J_m * (T_m - gamma * F_cl - T_f)
