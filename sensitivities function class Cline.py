import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import math

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
        km, k1, fc, fv, Ts = theta
        #when there is clamp force, x1>0
        if x1 > 0 and x2 > self.epsilon:
            dx2 = (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - (1 / self.J) * (fc + fv * x2)
            # print('1')
        elif x1 > 0 and x2 < -self.epsilon:
            dx2 = (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - (1 / self.J) * (-fc + fv * x2)
            # print('2')
        elif x1 > 0 and abs(x2) <= self.epsilon:
            if km * u - self.gamma * k1 * x1 > Ts: #overcome the maximum static friction
                dx2 = (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - Ts / self.J
                # print('3')
            else: #lockup
                dx2 =  (km / self.J) * u - (self.gamma * k1 / self.J) * x1 - ((km / self.J) * u - (self.gamma * k1 / self.J) * x1)
                # print('4')
        #when there is no clamp force, x1<=0
        elif x1 <= 0 and x2 > self.epsilon:
            dx2 = (km / self.J) * u - (1 / self.J) * (fc + fv * x2)
            # print('5')
        elif x1 <= 0 and x2 < -self.epsilon:
            dx2 = (km / self.J) * u - (1 / self.J) * (-fc + fv * x2)
            # print('6')
        elif x1 <= 0 and abs(x2) <= self.epsilon:
            if km * u > Ts: #overcome the maximum static friction
                dx2 = (km / self.J) * u - Ts / self.J
                # print('7')
            else: #lockup
                dx2 = (km / self.J) * u - (km / self.J) * u
                # print('8')
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
        df1_dx = [0, 1] #dx1 = x2
 
        if x1 > 0 and x2 > self.epsilon:
            df2_dx = [-self.gamma * self.k1 / self.J, -1 / self.J * self.fv]
        elif x1 > 0 and x2 < -self.epsilon:
            df2_dx = [-self.gamma * self.k1 / self.J, -1 / self.J * self.fv]
        elif x1 > 0 and abs(x2) <= self.epsilon:
            if self.km * u - self.gamma * self.k1 * x1 > self.Ts: #overcome the maximum static friction
                df2_dx = [-self.gamma * self.k1 / self.J, 0]
            else: #lockup
                df2_dx = [0, 0]
        #when there is no clamp force, x1<=0
        elif x1 <= 0 and x2 > self.epsilon:
            df2_dx = [0, -1 / self.J * self.fv]
        elif x1 <= 0 and x2 < -self.epsilon:
            df2_dx = [0, -1 / self.J * self.fv]
        elif x1 <= 0 and abs(x2) <= self.epsilon:
            if self.km * u > self.Ts: #overcome the maximum static friction
                df2_dx = [0, 0]
            else: #lockup
                df2_dx = [0, 0]
        return np.array([df1_dx, df2_dx])
    
    def jacobian_f_tf(self, x, u):
        x_tensor = tf.constant(x, dtype=tf.float32)
        u_tensor = tf.constant(u, dtype=tf.float32)
        theta_tensor = self.theta_tensor
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            f_x = self.f(x_tensor, u_tensor, theta_tensor)
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
        output: df/dtheta_norm = df/dtheta @ dtheta/dtheta_norm
        """
        x_tensor = tf.constant(x, dtype=tf.float32)
        u_tensor = tf.constant(u, dtype=tf.float32)
        theta_tensor = self.theta_tensor
        with tf.GradientTape() as tape:
            tape.watch(theta_tensor)
            f_x = self.f(x_tensor, u_tensor, theta_tensor)
        jacobian_df_dtheta = tape.jacobian(f_x, theta_tensor)
        jacobian_dtheta_dtheta_norm = tf.linalg.diag([self.km, self.k1, self.fc, self.fv, self.Ts])
        jacobian_df_dtheta_norm = np.array(tf.matmul(jacobian_df_dtheta, jacobian_dtheta_dtheta_norm))
        return jacobian_df_dtheta_norm

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
        elif x1 > 0 and x2 < -self.epsilon:
            df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
            df_dk1_nom = self.k1 * np.array([0, -self.gamma * x1 / self.J]).reshape(2,1)
            df_dfc_nom = self.fc * np.array([0, 1 / self.J]).reshape(2,1)
            df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
            df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
            # dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (-self.fc + self.fv * x2)
        elif x1 > 0 and abs(x2) <= self.epsilon:
            if self.km * u - self.gamma * self.k1 * x1 > self.Ts: #overcome the maximum static friction
                df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, -self.gamma * x1 / self.J]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, -1 / self.J]).reshape(2,1) 
                # dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - self.Ts / self.J
            else: #lockup
                df_dkm_nom = self.km * np.array([0, 0]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
                # dx2 = 0
        #when there is no clamp force, x1<=0
        elif x1 <= 0 and x2 > self.epsilon:
            df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
            df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
            df_dfc_nom = self.fc * np.array([0, -1 / self.J]).reshape(2,1)
            df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
            df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
            # dx2 = (self.km / self.J) * u - (1 / self.J) * (self.fc + self.fv * x2)
        elif x1 <= 0 and x2 < -self.epsilon:
            df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
            df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
            df_dfc_nom = self.fc * np.array([0, 1 / self.J]).reshape(2,1)
            df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
            df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
            # dx2 = (self.km / self.J) * u - (1 / self.J) * (-self.fc + self.fv * x2)
        elif x1 <= 0 and abs(x2) <= self.epsilon:
            if self.km * u > self.Ts: #overcome the maximum static friction
                df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, -1 / self.J]).reshape(2,1) 
                # dx2 = (self.km / self.J) * u - self.Ts / self.J
            else: #lockup
                df_dkm_nom = self.km * np.array([0, 0]).reshape(2,1)
                df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
                df_dfc_nom = self.fc * np.array([0, 0]).reshape(2,1)
                df_dfv_nom = self.fv * np.array([0, 0]).reshape(2,1)
                df_dTs_nom = self.Ts * np.array([0, 0]).reshape(2,1) 
                # dx2 = 0
        return np.concatenate((df_dkm_nom, df_dk1_nom, df_dfc_nom, df_dfv_nom, df_dTs_nom), axis=1)
    
    def sensitivity_x(self, J_f, df_dtheta, chi):
        """
        Define the sensitivity dx/dtheta with recursive algorithm
        chi(k+1) = chi(k) + dt * (J_x * chi(k) + df_dtheta)
        output: chi(k+1)
        """
        chi = chi + self.dt * (np.dot(J_f, chi) + df_dtheta)
        return chi

    def sensitivity_y(self, chi, J_h):
        """
        Define the sensitivity dy/dtheta
        dh_dtheta(k) = J_h * chi(k)
        output: dh_dtheta(k)
        """
        dh_dtheta = np.dot(J_h, chi)
        return dh_dtheta
    
    def fisher_info_matrix(self, dh_dtheta, R=1):
        """
        Define the fisher infomation matrix M
        dh_dtheta(k) = J_h * chi(k)
        output: fi_info
        """
        return np.dot(np.dot(dh_dtheta.reshape(-1,1), 1/R), dh_dtheta.reshape(1,-1))

# Initial state
x0 = np.array([0.0, 0.0])
chi = np.zeros((2,5))
fi_info = np.zeros((5,5))
det_T = 0.001
theta = np.array([21.7e-03, 23.04, 10.37e-3, 2.16e-5, 1.2 * (10.37e-3 + 2.16e-5 * 0.5)]) #[self.km, self.k1, self.fc, self.fv, self.Ts]

x0_values = []
x1_values = []
time_values = []
det_fi_values = []

fi_matrix = FI_matrix()
T = time.time()
x = x0

for k in range(350): #350 = 0.35s
    u = 1 + 1 * math.sin(2*math.pi*k/200 - math.pi/2)
    # u = 1.5
    dx = fi_matrix.f(x, u, theta)
    x = x + det_T * dx
    x0_values.append(x[0])
    x1_values.append(x[1])
    time_values.append((k+1) * det_T)
    # J_f_tf = fi_matrix.jacobian_f_tf(x, u)
    J_f = fi_matrix.jacobian_f(x, u)
    J_h = fi_matrix.jacobian_h(x)
    # df_theta_tf = fi_matrix.df_dtheta_tf(x, u)
    df_theta = fi_matrix.df_dtheta(x, u)
    chi = fi_matrix.sensitivity_x(J_f, df_theta, chi)
    dh_theta = fi_matrix.sensitivity_y(chi, J_h)
    fi_info_new = fi_matrix.fisher_info_matrix(dh_theta)
    fi_info += fi_info_new

    # if np.allclose(J_f, J_f_tf) == False:
    #     print('wrong with Jacobian caculation')
    # if np.allclose(df_theta, df_theta_tf) == False:
    #     print('wrong with Jacobian_theta caculation')


    C = np.linalg.eigvals(fi_info)
    for i in range(len(C)):
        if C[i] < 0:
            print(k, C[i])
    det_fi = np.linalg.det(fi_info)
    det_fi_values.append(det_fi)

print(fi_info)
print('det is', np.linalg.det(fi_info))
print('log det is', -np.log(np.linalg.det(fi_info)))

# save as csv
filename = 'output_3sin.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x0', 'x1', 'time', 'det_fi'])
    for x0, x1, time_value, det_fi_value in zip(x0_values, x1_values, time_values, det_fi_values):
        writer.writerow([x0, x1, time_value, det_fi_value])


# plt function
plt.subplot(4, 1, 1)
plt.plot(time_values, x0_values, label='x0')
plt.xlabel('Time (s)')
plt.ylabel('x0')
plt.title('x0 vs Time')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_values, x1_values, label='x1')
plt.xlabel('Time (s)')
plt.ylabel('x1')
plt.title('x1 vs Time')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time_values, det_fi_values, label='det')
plt.xlabel('Time (s)')
plt.ylabel('det')
plt.title('det vs Time')
plt.legend()

# plt.subplot(4, 1, 4)
# plt.plot(time_values, det_fi_newvalues, label='det_new')
# plt.xlabel('Time (s)')
# plt.ylabel('det_new')
# plt.title('det_new vs Time')
# plt.legend()

plt.tight_layout()
plt.savefig('3sin_350.png')
plt.show()