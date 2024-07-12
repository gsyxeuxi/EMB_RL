import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

class FI_matrix(object):

    def __init__(self) -> None:
        # Define parameters
        self.J = tf.constant(4.624e-06, dtype=tf.float32)  # Moment of inertia
        self.km = tf.constant(21.7e-03, dtype=tf.float32)  # Motor constant
        self.gamma = tf.constant(1.889e-05, dtype=tf.float32)  # Proportional constant
        self.k1 = tf.constant(23.04, dtype=tf.float32)  # Elasticity constant
        self.fc = tf.constant(10.37e-3, dtype=tf.float32)  # Coulomb friction coefficient
        self.epsilon = tf.constant(0.5, dtype=tf.float32)  # Zero velocity bound [rad/s]
        self.fv = tf.constant(2.16e-5, dtype=tf.float32)  # Viscous friction coefficient
        self.Ts = tf.constant(1.2 * (self.fc + self.fv * self.epsilon), dtype=tf.float32) #Static friction torque
        self.dt = 0.001
        self.theta_tensor = tf.stack([self.km, self.k1, self.fc, self.fv, self.Ts])
        print(self.theta_tensor)
        
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
        dx1 = x2
        self.km, self.k1, self.fc, self.fv, self.Ts = theta
        #when there is clamp force, x1>0
        if x1 > 0 and x2 > self.epsilon:
            dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (self.fc + self.fv * x2)
        if x1 > 0 and x2 < -self.epsilon:
            dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (-self.fc + self.fv * x2)
        if x1 > 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 > self.Ts: #overcome the maximum static friction
                dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - self.Ts
            else: #lockup
                dx2 = 0
        #when there is no clamp force, x1<=0
        if x1 <= 0 and x2 > self.epsilon:
            dx2 = (self.km / self.J) * u - (1 / self.J) * (self.fc + self.fv * x2)
        if x1 <= 0 and x2 < -self.epsilon:
            dx2 = (self.km / self.J) * u - (1 / self.J) * (-self.fc + self.fv * x2)
        if x1 <= 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u > self.Ts: #overcome the maximum static friction
                dx2 = (self.km / self.J) * u - self.Ts
            else: #lockup
                dx2 = 0
        # return np.array([dx1, dx2])
        return tf.convert_to_tensor([dx1, dx2], dtype=tf.float32)

    def h(self, x):
        """
        Define the output equation
        y = h(x)
        output:
            y: motor position
        """
        return x[0]

    def jacobian_f(self, x, u):
        """
        Define the Jacobian matrix of function f, J_f
        dx/dt = f(x, u, theta)
        output: J_f
        """
        x1, x2 = x
        df1_dx = [0, 1]
 
        if x1 > 0 and x2 > self.epsilon:
            df2_dx = [-self.gamma * self.k1 / self.J, -1 / self.J * self.fv]
        if x1 > 0 and x2 < -self.epsilon:
            df2_dx = [-self.gamma * self.k1 / self.J, -1 / self.J * self.fv]
        if x1 > 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 > self.Ts: #overcome the maximum static friction
                df2_dx = [-self.gamma * self.k1 / self.J, 0]
            else: #lockup
                df2_dx = [0, 0]
        #when there is no clamp force, x1<=0
        if x1 <= 0 and x2 > self.epsilon:
            df2_dx = [0, -1 / self.J * self.fv]
        if x1 <= 0 and x2 < -self.epsilon:
            df2_dx = [0, -1 / self.J * self.fv]
        if x1 <= 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u > self.Ts: #overcome the maximum static friction
                df2_dx = [0, 0]
            else: #lockup
                df2_dx = [0, 0]
        return np.array([df1_dx, df2_dx])
    
    def jacobian_f_tf(self, x, u):
        x_tensor = tf.constant(x, dtype=tf.float32)
        u_tensor = tf.constant(u, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            f_x = self.f(x_tensor, u_tensor)
        jacobian_matrix = np.array(tape.jacobian(f_x, x_tensor))
        return jacobian_matrix

    def jacobian_h(self, x):
        """
        Define the Jacobian matrix of function h, J_h
        y = h(x)
        output: J_h
        """
        x1, x2 = x
        dh_dx1 = 1
        dh_dx2 = 0
        return np.array([dh_dx1, dh_dx2])

    def df_dtheta_tf(self, x, u):
        """
        Define the matrix of df_dtheta with each parameter
        dx/dt = f(x, u, theta)
        output: df/dtheta
        """
        x_tensor = tf.constant(x, dtype=tf.float32)
        u_tensor = tf.constant(u, dtype=tf.float32)
        theta_tensor = self.theta_tensor
        with tf.GradientTape() as tape:
            tape.watch(theta_tensor)
            f_x = self.f(x_tensor, u_tensor, theta_tensor)
        jacobian_df_dtheta = np.array(tape.jacobian(f_x, theta_tensor))
        return jacobian_df_dtheta

    def df_dtheta(self, x, u):
        """
        Define the matrix of df_dtheta with each parameter
        dx/dt = f(x, u, theta)
        output: df/dtheta
        """
        x1, x2 = x
        #when there is clamp force, x1>0
        if x1 > 0 and x2 > self.epsilon:
            df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
            df_dk1_nom = self.k1 * np.array([0, -self.gamma * x1 / self.J]).reshape(2,1)
            df_dfc_nom = self.fc * np.array([0, -1 / self.J]).reshape(2,1)
            df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
            df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
            # dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (self.fc + self.fv * x2)
        if x1 > 0 and x2 < -self.epsilon:
            df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
            df_dk1_nom = self.k1 * np.array([0, -self.gamma * x1 / self.J]).reshape(2,1)
            df_dfc_nom = self.fc * np.array([0, 1 / self.J]).reshape(2,1)
            df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
            df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
            # dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (-self.fc + self.fv * x2)
        if x1 > 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 > self.Ts: #overcome the maximum static friction
                df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, -self.gamma * x1 / self.J]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, -1]).reshape(2,1) 
                # dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - self.Ts
            else: #lockup
                df_dkm_nom = self.km * np.array([0, 0]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
                # dx2 = 0
        #when there is no clamp force, x1<=0
        if x1 <= 0 and x2 > self.epsilon:
            df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
            df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
            df_dfc_nom = self.fc * np.array([0, -1 / self.J]).reshape(2,1)
            df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
            df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
            # dx2 = (self.km / self.J) * u - (1 / self.J) * (self.fc + self.fv * x2)
        if x1 <= 0 and x2 < -self.epsilon:
            df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
            df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
            df_dfc_nom = self.fc * np.array([0, 1 / self.J]).reshape(2,1)
            df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
            df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
            # dx2 = (self.km / self.J) * u - (1 / self.J) * (-self.fc + self.fv * x2)
        if x1 <= 0 and abs(x2) <= self.epsilon:
            if (self.km / self.J) * u > self.Ts: #overcome the maximum static friction
                df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, -1]).reshape(2,1) 
                # dx2 = (self.km / self.J) * u - self.Ts
            else: #lockup
                df_dkm_nom = self.km * np.array([0, 0]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
                # dx2 = 0
        return np.concatenate((df_dkm_nom, df_dk1_nom, df_dfc_nom, df_dfv_nom, df_dTs_nom), axis=1)
    
# Initial state
x0 = np.array([2.0, 4.0])
chi = np.zeros((2,5))
x0_values = []
x1_values = []
time_values = []
det_fi_values = []
det_fi_newvalues = []

fi_matrix = FI_matrix()
T = time.time()
det_T = 0.001
x = x0
u = 0.005
fi_info = np.zeros((5,5))
theta = np.array([21.7e-03, 23.04, 10.37e-3, 2.16e-5, 1.2 * (10.37e-3 + 2.16e-5 * 0.5)])
# theta = np.array([1, 1, 1, 1, 1])

for k in range(1): #350 = 0.35s
    u = 0.02
    dx = fi_matrix.f(x, u, theta)
    x = x + det_T * dx
    x0_values.append(x[0])
    x1_values.append(x[1])
    time_values.append(k * det_T)
    # J_f = fi_matrix.jacobian_f(x, u)
    # J_f_tf = fi_matrix.jacobian_f_tf(x, u)
    df_theta = fi_matrix.df_dtheta(x, u)
    df_theta_tf = fi_matrix.df_dtheta_tf(x, u)
    print(df_theta)
    print(df_theta_tf)

    # if np.allclose(J_f, J_f_tf) == False:
    #     print('wrong with Jacobian caculation')    