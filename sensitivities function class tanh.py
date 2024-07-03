import numpy as np
import time
from scipy.integrate import odeint

class FI_matrix(object):

    def __init__(self) -> None:
        # Define parameters
        self.J = 4.624e-06  # Moment of inertia
        self.km = 21.7e-03  # Motor constant
        self.gamma = 1.889e-05  # Proportional constant
        self.k1 = 23.04  # Elasticity constant
        self.fc = 10.37e-3  # Coulomb friction coefficient
        self.epsilon = 1000  # Excitation function coefficient
        self.fv = 2.16e-5  # Viscous friction coefficient
        # self.theta = self.km, self.k1, self.fc, self.fv, self.J, self.gamma, self.epsilon

    def f(self, x, u):
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
        if x1 > 0:
            dx2 = (self.km / self.J) * u - (self.gamma * self.k1 / self.J) * x1 - (1 / self.J) * (self.fc * np.tanh(self.epsilon * x2) + self.fv * x2)
        else: 
            dx2 = (self.km / self.J) * u - (1 / self.J) * (self.fc * np.tanh(self.epsilon * x2) + self.fv * x2)
        return np.array([dx1, dx2])

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
        if x1 > 0:
            df2_dx = [-self.gamma * self.k1 / self.J, -1 / self.J * (self.fc * self.epsilon * (1 / np.cosh(self.epsilon * x2))**2 + self.fv)]
        else:
            df2_dx = [0, -1 / self.J * (self.fc * self.epsilon * (1 / np.cosh(self.epsilon * x2))**2 + self.fv)]
        return np.array([df1_dx, df2_dx])

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

    def df_dtheta(self, x, u):
        """
        Define the matrix of df_dtheta with each parameter
        dx/dt = f(x, u, theta)
        output: df/dtheta
        """
        x1, x2 = x
        # df_dkm = np.array([0, 1 / self.J * u]).reshape(2,1)
        # df_dk1 = np.array([0, -self.gamma * x1 / self.J]).reshape(2,1)
        # df_dfc = np.array([0, -np.tanh(self.epsilon * x2) / self.J]).reshape(2,1)
        # df_dfv = np.array([0, -x2 / self.J]).reshape(2,1)
        # return np.array([df_dkm, df_dk1, df_dfc, df_dfv])
        df_dkm_nom = self.km * np.array([0, 1 / self.J * u]).reshape(2,1)
        if x1 > 0:
            df_dk1_nom = self.k1 * np.array([0, -self.gamma * x1 / self.J]).reshape(2,1)
        else:
            df_dk1_nom = self.k1 * np.array([0, 0]).reshape(2,1)
        df_dfc_nom = self.fc * np.array([0, -np.tanh(self.epsilon * x2) / self.J]).reshape(2,1)
        df_dfv_nom = self.fv * np.array([0, -x2 / self.J]).reshape(2,1)
        return np.array([df_dkm_nom, df_dk1_nom, df_dfc_nom, df_dfv_nom])
    

    def sensitivity_x(self, J_f, df_dtheta, chi):
        """
        Define the sensitivity dx/dtheta with recursive algorithm
        chi(k+1) = J_x * chi(k) + df_dtheta
        output: chi(k+1)
        """
        for i in range(len(df_dtheta)):
            chi[i] = np.dot(J_f, chi[i]) + df_dtheta[i]
        return chi

    def sensitivity_y(self, chi, J_h):
        """
        Define the sensitivity dy/dtheta
        dh_dtheta(k) = J_h * chi(k)
        output: dh_dtheta(k)
        """
        dh_dtheta = np.zeros((1, len(chi)))
        for i in range(len(chi)):
            dh_dtheta[0][i] = np.dot(J_h, chi[i]).item()
        return dh_dtheta
    
    def fisher_info_matrix(self, dh_dtheta, R=1):
        """
        Define the fisher infomation matrix M
        dh_dtheta(k) = J_h * chi(k)
        output: fi_info
        """
        return np.dot(np.dot(dh_dtheta.reshape(-1,1), 1/R), dh_dtheta)

# Initial state
x0 = np.array([0.0, 0.0])
chi = np.zeros((4,2,1))

fi_matrix = FI_matrix()

T = time.time()
det_T = 0.01
x = x0
fi_info = np.zeros((4,4))
for k in range(10):
    # u = k * det_T
    u = 0.001
    dx = fi_matrix.f(x, u)
    x = x + det_T * dx
    # print('x0 is', x[0])
    J_f = fi_matrix.jacobian_f(x, u)
    J_h = fi_matrix.jacobian_h(x)
    df_theta = fi_matrix.df_dtheta(x, u)
    chi = fi_matrix.sensitivity_x(J_f, df_theta, chi)
    dh_theta = fi_matrix.sensitivity_y(chi, J_h)
    fi_info += fi_matrix.fisher_info_matrix(dh_theta)
    print('det is', np.linalg.det(fi_info))
    print(-np.log(np.linalg.det(fi_info)))
    # if k > 1:
    #     print('det M-1', np.linalg.det(np.linalg.inv(fi_info)))
    #     print(-np.log(np.linalg.det(fi_info)))


