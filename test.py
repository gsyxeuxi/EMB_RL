import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import csv
import math
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients

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
        self.Ts = 1.2 * (self.fc + self.fv * self.epsilon) #Static friction torque
        self.dt = 0.001
        self.theta_tensor = tf.convert_to_tensor([self.km, self.k1, self.fc, self.fv, self.Ts])
        # self.theta_tensor = np.array([self.km, self.k1, self.fc, self.fv, self.Ts])

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
        x1, x2 = x
        
        km, k1, fc, fv, Ts = theta
        dx1 = x2
        #when there is clamp force, x1>0
        if x1 > 0 and x2 > self.epsilon:
            dx2 = (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - (1 / self.J) * (fc + fv * x2)
        elif x1 > 0 and x2 < -self.epsilon:
            dx2 = (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - (1 / self.J) * (-fc + fv * x2)
        elif x1 > 0 and abs(x2) <= self.epsilon:
            if (km / self.J) * u - (self.gamma * k1 / self.J) * x1 > Ts: #overcome the maximum static friction
                dx2 = (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - Ts
            else: #lockup
                dx2 = 0
        #when there is no clamp force, x1<=0
        elif x1 <= 0 and x2 > self.epsilon:
            dx2 = (km / self.J) * u - (1 / self.J) * (fc + fv * x2)
        elif x1 <= 0 and x2 < -self.epsilon:
            dx2 = (km / self.J) * u - (1 / self.J) * (-fc + fv * x2)
        elif x1 <= 0 and abs(x2) <= self.epsilon:
            if (km / self.J) * u > Ts: #overcome the maximum static friction
                dx2 = (km / self.J) * u - Ts
            else: #lockup
                dx2 = 0
        dx2 = (km / self.J) * u - (1 / self.J) * (-fc + fv * x2)
        
        return tf.convert_to_tensor([dx1, dx2], dtype=tf.float32)

    def df_dtheta_tf(self, x, u):
        """
        Define the matrix of df_dtheta with each parameter
        dx/dt = f(x, u, theta)
        output: df/dtheta_norm = df/dtheta @ dtheta/dtheta_norm
        """
        x_tensor = tf.constant(x, dtype=tf.float32)
        u_tensor = tf.constant(u, dtype=tf.float32)
        theta_tensor = self.theta_tensor
        with tf.GradientTape() as tape:
            tape.watch(theta_tensor)
            f_x = self.f(x_tensor, u_tensor, theta_tensor)
        jacobian_df_dtheta = tape.jacobian(f_x, theta_tensor, unconnected_gradients=UnconnectedGradients.ZERO)
        jacobian_dtheta_dtheta_norm = tf.linalg.diag([self.km, self.k1, self.fc, self.fv, self.Ts])
        # jacobian_df_dtheta_norm = np.array(tf.matmul(jacobian_df_dtheta, jacobian_dtheta_dtheta_norm))
        return jacobian_df_dtheta
    
    def jacobian_f_tf(self, x, u):
        x_tensor = tf.constant(x, dtype=tf.float32)
        u_tensor = tf.constant(u, dtype=tf.float32)
        theta_tensor = self.theta_tensor
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            f_x = self.f(x_tensor, u_tensor, theta_tensor)
        jacobian_matrix = np.array(tape.jacobian(f_x, x_tensor))
        return jacobian_matrix
    
    # this works 
    def jacobian(self, x, u):
        # x_tensor = tf.constant(x, dtype=tf.float32)
        # u_tensor = tf.constant(u, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(self.theta_tensor)
            f_x = self.f(x, u, self.theta_tensor)
        jacobian_matrix = np.array(tape.jacobian(f_x, x))
        jacobian_df_dtheta = tape.jacobian(f_x, self.theta_tensor, UnconnectedGradients.ZERO)
        return jacobian_matrix, jacobian_df_dtheta

test = FI_matrix()

x= [0.0, 0.0]
det_T = 0.001
theta = np.array([21.7e-03, 23.04, 10.37e-3, 2.16e-5, 1.2 * (10.37e-3 + 2.16e-5 * 0.5)]) 
for i in range(10):
    u = 0.05+0.01*i
    # print('u =', u)
    # u = 0.03 + 0.03 * math.sin(2*math.pi*k/200 - math.pi)
    dx = test.f(x, u, theta)
    x = x + det_T * dx

    df_theta = test.df_dtheta_tf(x, u)
    df_dx = test.jacobian_f_tf(x,u)
    print(test.jacobian(x,u)[0])
    print(df_dx)
    print('*******************')
    print(test.jacobian(x,u)[1])
    print(df_theta)
    print('*******************')
