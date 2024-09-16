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
        # self.theta_tensor = tf.convert_to_tensor([self.km, self.k1, self.fc, self.fv], dtype=tf.float64)
        self.theta_tensor = tf.convert_to_tensor([self.fv], dtype=tf.float64)
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
        km = self.km
        k1 = self.k1
        fc = self.fc
        fv = theta
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
    
    def fisher_info_matrix(self, dh_dtheta, R=0.05):
        """
        Define the fisher infomation matrix M
        M = dh_dtheta.T * (1/R) * dh_dtheta
        output: fi_info
        """
        return tf.matmul(dh_dtheta, dh_dtheta, True) * (1/R)

    def symmetrie_test(self, x):
        """
        Test if a matrix is symmetrie
        """
        x = np.array(x)
        y = (x + np.transpose(x)) / 2
        if np.array_equal(x, y) == False:
            print('not symmetrie matrix')

        try:
            np.linalg.cholesky(x)
            return True
        except np.linalg.LinAlgError:
            return False

    def fi_matrix_scale(self, x, u, scale_factor, fi_info_previous_scale):
        jacobian = self.jacobian(x, u)
        J_f, df_theta = jacobian
        J_h = self.jacobian_h(x)
        self.chi = self.sensitivity_x(J_f, df_theta, self.chi)
        dh_theta = self.sensitivity_y(chi, J_h)
        fi_info_new = self.fisher_info_matrix(dh_theta)
        fi_info_new_scale = fi_info_new * scale_factor
        fi_info_scale = fi_info_previous_scale + fi_info_new_scale
        self.symmetrie_test(fi_info_scale)
        return fi_info_scale



# Initial state
x_0 = tf.Variable([0.0, 0.0], dtype=tf.float64)

chi = tf.convert_to_tensor(np.zeros((2,1)), dtype=tf.float64)
fi_info = tf.convert_to_tensor(np.eye(1) * 1e-6, dtype=tf.float64)
det_T = 0.001 #Time step
theta = tf.constant([2.16e-5], dtype=tf.float64) #[self.km, self.k1, self.fc, self.fv]

fi_matrix = FI_matrix()
x = x_0
pi = tf.constant(math.pi, dtype=tf.float64)
scale_factor = 1
scale_factor_previous = 1
det_init = tf.linalg.det(fi_info)
fi_info_scale = fi_info * scale_factor
fi_info_previous_scale = fi_info_scale
det_previous_scale = tf.linalg.det(fi_info_previous_scale)
log_det_previous_scale = tf.math.log(det_previous_scale)
total_reward_scale = log_det_previous_scale

'''
start the simulation
'''
for k in range(5): #350 = 0.35s
    # case1: sinus input
    # u = tf.Variable(2 + 2 * tf.math.sin(2*pi*k/100 - pi/2), dtype=tf.float64)
    u = 2
    dx = fi_matrix.f(x, u, theta)
    x = x + det_T * dx
    jacobian = fi_matrix.jacobian(x, u)
    J_f = jacobian[0]
    J_h = fi_matrix.jacobian_h(x)
    df_theta = jacobian[1]
    chi = fi_matrix.sensitivity_x(J_f, df_theta, chi)
    dh_theta = fi_matrix.sensitivity_y(chi, J_h)
    fi_info_new = fi_matrix.fisher_info_matrix(dh_theta)
    print(fi_info_new)
    fi_info_new_scale = fi_info_new * scale_factor
    fi_info_scale = fi_info_previous_scale + fi_info_new_scale
    fi_info += fi_info_new
    fi_matrix.symmetrie_test(fi_info_scale)
 
    det_fi_scale = tf.linalg.det(fi_info_scale)
    log_det_scale = tf.math.log(det_fi_scale)
    step_reward_scale = log_det_scale - log_det_previous_scale
    print(step_reward_scale)
    total_reward_scale = total_reward_scale + step_reward_scale
    # calculate for the next step
    scale_factor = (det_init / det_fi_scale) ** (1/4)
    fi_info_previous_scale = fi_info_scale * (scale_factor / scale_factor_previous)
    log_det_previous_scale = np.linalg.slogdet(fi_info_previous_scale)[1]
    scale_factor_previous = scale_factor


# The calculation of the det or log det scale is not meaningful because the scale is only used for reward calculation and has no exact meaning in physics.
print('det sacle is', np.linalg.det(fi_info_scale))
print('log det scale is', np.log(np.linalg.det(fi_info_scale)))
print("total_reward_scale", total_reward_scale)
