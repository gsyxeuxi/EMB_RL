import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
import time
import csv
import math

"""
I_m_cont = 4.87[A]
I_m_max = 6[A]
x1_max =
"""

class FI_matrix(object):

    def __init__(self) -> None:
        # Define parameters
        self.J = 4.624e-06  # Moment of inertia
        self.km = 21.7e-03  # Motor constant
        self.gamma = 1.889e-05  # Proportional constant
        self.k1 = 23.04  # Elasticity constant
        self.fc = 10.37e-3  # Coulomb friction coefficient
        self.epsilon = 0.5  # Zero velocity bound [rad/s]
        self.fv = 2.16e-5  # Viscous friction coefficient
        self.Ts = 1.2 * (self.fc + self.fv * self.epsilon) # Static friction torque
        self.dt = 0.001
        self.theta_tensor = tf.convert_to_tensor([self.km, self.k1, self.fc, self.fv, self.Ts], dtype=tf.float64)

    @tf.function
    def f(self, x, u, theta):
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
        x1, x2 = tf.unstack(x)
        km, k1, fc, fv, Ts = tf.unstack(theta)
        dx1 = x2
        def clamp_force_condition():
            def moving():
                return (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - (1 / self.J) * (fc * tf.sign(x2) + fv * x2)
            
            def not_moving():
                return tf.cond(
                    km * u - self.gamma * k1 * x1 > Ts,
                    lambda: (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - Ts / self.J,
                    lambda: tf.constant(0.0, dtype=tf.float64)
                )
            return tf.cond(tf.abs(x2) > self.epsilon, moving, not_moving)
        
        def no_clamp_force_condition():
            def moving():
                return (km / self.J) * u - (1 / self.J) * (fc * tf.sign(x2) + fv * x2)
            
            def not_moving():
                return tf.cond(
                    km * u > Ts,
                    lambda: (km / self.J) * u - Ts / self.J,
                    lambda: tf.constant(0.0, dtype=tf.float64)
                )
            return tf.cond(tf.abs(x2) > self.epsilon, moving, not_moving)
        
        dx2 = tf.cond(x1 > 0, clamp_force_condition, no_clamp_force_condition)
        return tf.convert_to_tensor([dx1, dx2], dtype=tf.float64)

    def h(self, x):
        """
        Define the output equation
        y = h(x)
        output:
            y: motor position
        """
        return x[0]

    def jacobian_h(self, x):
        """
        Define the Jacobian matrix of function h, J_h
        y = h(x)
        output: J_h
        """
        x1, x2 = x
        dh_dx1 = 1
        dh_dx2 = 0
        return tf.convert_to_tensor([[dh_dx1, dh_dx2]], dtype=tf.float64)
    
    @tf.function
    def jacobian(self, x, u):
        """
        Define the Jacobian matrix of function f, J_f,
        and the matrix of df_dtheta with each parameter
        dx/dt = f(x, u, theta)
        output: J_f, df/dtheta_norm = df/dtheta @ dtheta/dtheta_norm
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(self.theta_tensor)
            f_x = self.f(x, u, self.theta_tensor)
        jacobian_df_dx = tape.jacobian(f_x, x)
        jacobian_df_dtheta = tape.jacobian(f_x, self.theta_tensor, UnconnectedGradients.ZERO)
        jacobian_dtheta_dtheta_norm = tf.linalg.diag(np.array(self.theta_tensor))
        jacobian_df_dtheta_norm = tf.matmul(jacobian_df_dtheta, jacobian_dtheta_dtheta_norm)
        return jacobian_df_dx, jacobian_df_dtheta_norm

    def sensitivity_x(self, J_f, df_dtheta, chi):
        """
        Define the sensitivity dx/dtheta with recursive algorithm
        chi(k+1) = chi(k) + dt * (J_x * chi(k) + df_dtheta)
        output: chi(k+1)
        """
        chi = chi + self.dt * (tf.matmul(J_f, chi) + df_dtheta)
        return chi  

    def sensitivity_y(self, chi, J_h):
        """
        Define the sensitivity dy/dtheta
        dh_dtheta(k) = J_h * chi(k)
        output: dh_dtheta(k)
        """
        dh_dtheta = tf.matmul(J_h, chi)
        return dh_dtheta
    
    def fisher_info_matrix(self, dh_dtheta, R=1):
        """
        Define the fisher infomation matrix M
        dh_dtheta(k) = J_h * chi(k)
        output: fi_info
        """
        return tf.matmul(tf.multiply(dh_dtheta, 1/R), dh_dtheta, True)
        # return np.dot(np.dot(np.array(dh_dtheta).reshape(-1,1), 1/R), np.array(dh_dtheta).reshape(1,-1))

# Initial state
x_0 = tf.constant([0.0, 0.0], dtype=tf.float64)

chi = tf.convert_to_tensor(np.zeros((2,5)), dtype=tf.float64)
fi_info = tf.convert_to_tensor(np.zeros((5,5)), dtype=tf.float64)
fi_info_10 = tf.convert_to_tensor(np.zeros((5,5)), dtype=tf.float64)
det_T = 0.001
theta = tf.constant([21.7e-03, 23.04, 10.37e-3, 2.16e-5, 1.2 * (10.37e-3 + 2.16e-5 * 0.5)], dtype=tf.float64) #[self.km, self.k1, self.fc, self.fv, self.Ts]

x0_values = []
x1_values = []
time_values = []
time_values_10 = []
det_fi_values = []
det_fi_values_10 = []
log_dets = []
log_dets_10 = []
sensitivity_y = []

fi_matrix = FI_matrix()
T = time.time()
x = x_0
pi = tf.constant(math.pi, dtype=tf.float64)
t = time.time()

for k in range(350): #350 = 0.35s
    t = time.time()
    u = tf.constant(1 + 1 * tf.math.sin(2*pi*k/200 - pi/2), dtype=tf.float64)
    dx = fi_matrix.f(x, u, theta)
    x = x + det_T * dx
    x0_values.append(x[0])
    x1_values.append(x[1])
    time_values.append((k+1) * det_T)
    jacobian = fi_matrix.jacobian(x, u)
    J_f = jacobian[0]
    J_h = fi_matrix.jacobian_h(x)
    df_theta = jacobian[1]
    chi = fi_matrix.sensitivity_x(J_f, df_theta, chi)
    dh_theta = fi_matrix.sensitivity_y(chi, J_h)

    fi_info_new = fi_matrix.fisher_info_matrix(dh_theta)

    # det_previous = det(fi_info)
    # fi_info += fi_info_new
    # det_new = det(fi_info)
    # reward_intermediate = det_new - det_old
    # ... at the end of episode
    # reward_final = det(fi_info)

    fi_info += fi_info_new

    if k % 70 == 0:
        print(fi_info_10)
        det_fi_10 = tf.linalg.det(fi_info_10)
        log_dets_10.append(-tf.math.log(det_fi_10))
        det_fi_values_10.append(det_fi_10)
        time_values_10.append((k+1) * det_T)
        fi_info_10 = fi_info_new
    
    else:
        fi_info_10 += fi_info_new
    det_fi = tf.linalg.det(fi_info)
    det_fi_values.append(det_fi)
    log_dets.append(-tf.math.log(det_fi))
    sensitivity_y.append(dh_theta)
    print(time.time()-t)
print('det is', np.linalg.det(fi_info))
print('log det is', -np.log(np.linalg.det(fi_info)))

# save as csv
filename = 'output_1sinus_v2.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x0', 'x1', 'time', 'det_fi', 'log_det', 'sensitivity_y'])
    for x0, x1, time_value, det_fi_value, log_det, dh_theta in zip(x0_values, x1_values, time_values, det_fi_values, log_dets, sensitivity_y):
        writer.writerow([x0, x1, time_value, det_fi_value, log_det, dh_theta])
    writer.writerow(['det_fi_10', 'log_det_10'])
    for det_fi_value_10, log_det_10 in zip(det_fi_values_10, log_dets_10):
        writer.writerow([det_fi_value_10, log_det_10])


# plt function
plt.subplot(2, 3, 1)
plt.plot(time_values, x0_values, label='x0')
plt.xlabel('Time (s)')
plt.ylabel('x0')
plt.title('x0 vs Time')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(time_values, x1_values, label='x1')
plt.xlabel('Time (s)')
plt.ylabel('x1')
plt.title('x1 vs Time')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(time_values, det_fi_values, label='det')
plt.xlabel('Time (s)')
plt.ylabel('det')
plt.title('det vs Time')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(time_values_10, det_fi_values_10, label='det_10')
plt.xlabel('Time (s)')
plt.ylabel('det_10')
plt.title('det_10 vs Time')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(time_values, log_dets, label='log_det')
plt.xlabel('Time (s)')
plt.ylabel('log_det')
plt.title('log_det vs Time')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(time_values_10, log_dets_10, label='log_det_10')
plt.xlabel('Time (s)')
plt.ylabel('log_det_10')
plt.title('log_det_10 vs Time')
plt.legend()

plt.tight_layout()
plt.savefig('1sinus_v2.png')
plt.show()   
print("111", time.time()-t)
print('Finished')