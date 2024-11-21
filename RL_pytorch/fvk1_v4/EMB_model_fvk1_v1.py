import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.normal import Normal
import math
import time


"""
add:
h(x) = [[x1], [x2]]
self.R_inv = torch.tensor([[1e6, 0],[0, 1]], dtype=torch.float64)
"""


class FI_matrix(object):
    
    def __init__(self):
        # Define parameters
        self.J = 4.624e-06  # Moment of inertia
        self.km = 21.7e-03  # Motor constant
        self.gamma = 1.889e-05  # Proportional constant
        # self.k1 = 23.04  # Elasticity constant
        self.fc = 10.37e-3  # Coulomb friction coefficient
        self.epsilon = 0.5  # Zero velocity bound [rad/s]
        self.fv = torch.tensor([2.16e-5], dtype=torch.float64)  # Viscous friction coefficient
        self.dt = 0.001
        self.R_inv = torch.tensor([[1e6, 0],[0, 1]], dtype=torch.float64) # Invers matrix of the sensor noise

    def f(self, x, u, theta):
        x1, x2 = x[0], x[1]
        fv, k1 = theta[0], theta[1]
        km = self.km
        fc = self.fc
        dx1 = x2
        Tm = km * u
        Tl = self.gamma * k1 * torch.max(x1, torch.tensor(0.0, dtype=torch.float64))
        Tf = fc * torch.sign(x2) + fv * x2 # variant 1
        # Tf = fc * torch.tanh(1000 * x2) + fv * x2 # variant 2
        dx2 = (Tm - Tl - Tf) / self.J
        return torch.stack([dx1, dx2])

    def h(self, x):
        """
        Define the output equation
        y = h(x)
        output:
            y: motor position
        """
        # return x[0]
        return torch.stack([x[0], x[1]])

    def jacobian_h(self, x):
        """
        Define the Jacobian matrix of function h, J_h
        y = h(x)
        output: J_h
        """
        dh_dx1 = torch.tensor([1/100, 0], dtype=torch.float64)
        dh_dx2 = torch.tensor([0, 1/500], dtype=torch.float64)
        return torch.stack([dh_dx1, dh_dx2])

    def jacobian(self, x, u, theta):
        """
        Define the Jacobian matrix of function f, J_f,
        and the matrix of df_dtheta with each parameter
        dx/dt = f(x, u, theta)
        output: J_f, df/dtheta_norm = df/dtheta @ dtheta/dtheta_norm
        """
        # x = torch.autograd.Variable(x, requires_grad=True)
        # theta = torch.autograd.Variable(self.fv, requires_grad=True)
        x = x.clone().detach().requires_grad_(True)
        # theta = self.fv.clone().detach().requires_grad_(True)
        theta1 = theta.clone().detach().requires_grad_(True)
        f_x = self.f(x, u, theta1)

        jacobian_df_dx = []
        jacobian_df_dtheta = []
        for i in range(f_x.size(0)):
            grad_x = torch.autograd.grad(f_x[i], x, retain_graph=True, create_graph=True)[0]
            grad_theta = torch.autograd.grad(f_x[i], theta1, retain_graph=True, create_graph=True)[0]
            jacobian_df_dx.append(grad_x)
            jacobian_df_dtheta.append(theta * grad_theta)
        jacobian_df_dx = torch.stack(jacobian_df_dx, dim=0)
        jacobian_df_dtheta = torch.stack(jacobian_df_dtheta, dim=0)

        # jacobian_dtheta_dtheta_norm = torch.diag(self.fv)
        # jacobian_df_dtheta_norm = torch.matmul(jacobian_df_dtheta.view(-1, 1), jacobian_dtheta_dtheta_norm))

        return jacobian_df_dx, jacobian_df_dtheta

    def sensitivity_x(self, J_f, df_dtheta, chi):
        """
        Define the sensitivity dx/dtheta with recursive algorithm
        chi(k+1) = chi(k) + dt * (J_x * chi(k) + df_dtheta)
        output: chi(k+1)
        """
        chi = chi + self.dt * (torch.matmul(J_f, chi) + df_dtheta)
        return chi

    def sensitivity_y(self, chi, J_h):
        """
        Define the sensitivity dy/dtheta
        dh_dtheta(k) = J_h * chi(k)
        output: dh_dtheta(k)
        """
        dh_dtheta = torch.matmul(J_h, chi)
        return dh_dtheta
    
    def fisher_info_matrix(self, dh_dtheta):
        """
        Define the fisher infomation matrix M
        M = dh_dtheta.T * (1/R) * dh_dtheta
        output: fi_info
        """
        return torch.matmul(torch.matmul(dh_dtheta.t(), self.R_inv), dh_dtheta)

    def symmetrie_test(self, x):
        """
        Test if a matrix is symmetric
        """
        x = x.detach().numpy()
        y = (x + np.transpose(x)) / 2
        if not np.array_equal(x, y):
            print('not a symmetric matrix')

        try:
            np.linalg.cholesky(x)
            return True
        except np.linalg.LinAlgError:
            return False

