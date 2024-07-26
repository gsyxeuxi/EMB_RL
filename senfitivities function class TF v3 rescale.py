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
        # self.Ts = 1.0 * (self.fc + self.fv * self.epsilon) # Static friction torque
        self.dt = 0.001
        self.theta_tensor = tf.convert_to_tensor([self.km, self.k1, self.fc, self.fv], dtype=tf.float64)

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
    @tf.function
    def f(self, x, u, theta):
        x1, x2 = tf.unstack(x)
        km, k1, fc, fv= tf.unstack(theta)
        dx1 = x2
        Tm = km * u
        Tl = self.gamma * k1 * tf.maximum(x1, 0.0)
        Tf = ((fc * tf.sign(x2) + fv * x2) / self.epsilon) * tf.minimum(tf.abs(x2), self.epsilon)
        dx2 = (Tm - Tl - Tf) / self.J
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
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
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

    def PSD_test(self, x):
        x = np.array(x)
        y = (x + np.transpose(x)) / 2
        if np.array_equal(x, y) == False:
            print('not symmetrie matrix')

        try:
            np.linalg.cholesky(x)
            return True
        except np.linalg.LinAlgError:
            return False

# Initial state
x_0 = tf.Variable([0.0, 0.0], dtype=tf.float64)

chi = tf.convert_to_tensor(np.zeros((2,4)), dtype=tf.float64)
fi_info = tf.convert_to_tensor(np.eye(4) * 1e-6, dtype=tf.float64)
det_T = 0.001
theta = tf.constant([21.7e-03, 23.04, 10.37e-3, 2.16e-5], dtype=tf.float64) #[self.km, self.k1, self.fc, self.fv]

x0_values = []
x1_values = []
time_values = []
det_fi_values = []
reward_values = []
log_det_values = []
# log_dets_10 = []
sensitivity_y = []

fi_matrix = FI_matrix()
T = time.time()
x = x_0
pi = tf.constant(math.pi, dtype=tf.float64)
t = time.time()

scale_factor = 1e-6
fi_info_scale = fi_info * scale_factor

det_previous = tf.linalg.det(fi_info)
log_det_previous = tf.math.log(det_previous)
det_previous_scale = tf.linalg.det(fi_info_scale)
log_det_previous_scale = tf.math.log(det_previous_scale)

total_reward = log_det_previous
total_reward_scale = log_det_previous

# calculate the determinat by Laplace expansion
def minor(matrix, i, j):
    """
    Returns the minor of the matrix excluding the i-th row and j-th column.
    """
    minor = np.delete(matrix, i, axis=0)
    minor = np.delete(minor, j, axis=1)
    return minor

def determinant(matrix):
    """
    Recursively calculates the determinant of a matrix.
    """
    # Base case for 2x2 matrix
    if matrix.shape == (2, 2):
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    det = 0
    for col in range(matrix.shape[1]):
        det += ((-1) ** col) * matrix[0, col] * determinant(minor(matrix, 0, col))
    return det


for k in range(350): #350 = 0.35s
    # print('step', k+1, '-----------------------------------------------')
    
    # #case1: sinus input
    # u = tf.Variable(1 + 1 * tf.math.sin(2*pi*k/200 - pi/2), dtype=tf.float64)
    # case2: slope
    u = tf.Variable(3/350 * k, dtype=tf.float64)
    # # case3: one sinus peroide
    # if k <= 200:
    #     u = tf.Variable(1 + 1 * tf.math.sin(2*pi*k/200 - pi/2), dtype=tf.float64)
    # else:
    #     u = tf.constant(0.0, dtype=tf.float64)
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
    sensitivity_y.append(dh_theta)

    fi_info_new = fi_matrix.fisher_info_matrix(dh_theta)
    fi_info_new_scale = fi_info_new * scale_factor
    fi_info += fi_info_new
    fi_info_scale += fi_info_new_scale
    fi_matrix.PSD_test(fi_info)
    fi_matrix.PSD_test(fi_info_scale)

    det_fi = tf.linalg.det(fi_info)

    det_fi_values.append(det_fi)
    np.linalg.cholesky(fi_info)
    log_det = tf.math.log(det_fi)
    log_det_values.append(log_det)

    det_fi_scale = tf.linalg.det(fi_info_scale)
    log_det_scale = tf.math.log(det_fi_scale)

    step_reward = log_det - log_det_previous
    reward_values.append(step_reward)
    total_reward = total_reward + step_reward

    step_reward_scale = log_det_scale - log_det_previous_scale
    total_reward_scale = total_reward_scale + step_reward_scale
    # scale_factor = tf.constant((1/det_fi)**(1/4), dtype=tf.float64)

    log_det_previous = log_det
    log_det_previous_scale = log_det_scale

    # print('step reward is:', step_reward)
    # print('step reward scale is:', step_reward_scale)

print('det is', np.linalg.det(fi_info))
print('log det is', np.log(np.linalg.det(fi_info)))
print('det sacle is', np.linalg.det(fi_info_scale))
print('log det scale is', np.log(np.linalg.det(fi_info_scale)))
print("total_reward", total_reward)
print("total_reward_scale", total_reward_scale)


# save as csv
# filename = 'output_0.5slop_v3.csv'

# with open(filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['x0', 'x1', 'time', 'det_fi', 'log_det', 'sensitivity_y'])
#     for x0, x1, time_value, det_fi_value, reward_value, log_det, dh_theta in zip(x0_values, x1_values, time_values, det_fi_values, reward_values, log_dets, sensitivity_y):
#         writer.writerow([x0, x1, time_value, det_fi_value, reward_value, log_det, dh_theta])



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
plt.plot(time_values, reward_values, label='det_10')
plt.xlabel('Time (s)')
plt.ylabel('Step rewards')
plt.title('Step rewards vs Time')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(time_values, log_det_values, label='log_det')
plt.xlabel('Time (s)')
plt.ylabel('log_det')
plt.title('log_det vs Time')
plt.legend()

# plt.subplot(2, 3, 6)
# plt.plot(time_values_10, log_dets_10, label='log_det_10')
# plt.xlabel('Time (s)')
# plt.ylabel('log_det_10')
# plt.title('log_det_10 vs Time')
# plt.legend()

plt.tight_layout()
# plt.savefig('0.5slop_v3.png')
plt.show()   
print("Running time is:", time.time()-t)
print('Finished')