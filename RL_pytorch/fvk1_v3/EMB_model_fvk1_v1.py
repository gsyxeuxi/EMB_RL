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
        # Tf = ((fc * torch.sign(x2) + fv * x2) / self.epsilon) * torch.min(torch.abs(x2), torch.tensor(self.epsilon, dtype=torch.float64)) # old wrong
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
        # dh_dx1 = torch.tensor([1.0, 0], dtype=torch.float64)
        # dh_dx2 = torch.tensor([0, 1.0], dtype=torch.float64)
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
        # jacobian_df_dtheta_norm = torch.matmul(jacobian_df_dtheta.view(-1, 1), jacobian_dtheta_dtheta_norm)

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

# # Initial state
# x_0 = torch.tensor([-0.2, 1.0], dtype=torch.float64)
# # x_0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
# chi = torch.zeros((2, 2), dtype=torch.float64)
# # fi_info = torch.eye(1, dtype=torch.float64) * 1e-6
# fi_info = torch.diag(torch.tensor([0.0, 0.0]))
# det_T = 0.001  # Time step
# theta = torch.tensor([2.16e-5, 23.04], dtype=torch.float64)

# fi_matrix = FI_matrix()
# x = x_0
# pi = torch.tensor(math.pi, dtype=torch.float64)
# scale_factor = 1.0
# scale_factor_previous = 1.0
# det_init = torch.det(fi_info)
# fi_info_scale = fi_info * scale_factor
# fi_info_previous_scale = fi_info_scale
# det_previous_scale = torch.det(fi_info_previous_scale)
# log_det_previous_scale = torch.log(det_previous_scale)
# total_reward_scale = log_det_previous_scale
# log_det_previous = torch.log(det_init)
# total_reward = 0

# # Start the simulation
# for k in range(400):  # 350 = 0.35s
#     # u = 3 + 3 * math.sin(2*pi*k/100 + pi/2)
#     u = 6
#     # if k > 150:
#     #     u = 0
#     dx = fi_matrix.f(x, u, theta)
#     x = x + det_T * dx
#     # x = torch.tensor([10.0, 10.0], dtype=torch.float64)
#     J_f, df_theta = fi_matrix.jacobian(x, u, theta)
#     J_h = fi_matrix.jacobian_h(x)
#     chi = fi_matrix.sensitivity_x(J_f, df_theta, chi)
#     # print('chi', chi)
#     dh_theta = fi_matrix.sensitivity_y(chi, J_h)

#     fi_info_new = fi_matrix.fisher_info_matrix(dh_theta)
#     # print('reward', fi_info_new)
#     fi_info_new_scale = fi_info_new * scale_factor
#     fi_info_scale = fi_info_previous_scale + fi_info_new_scale
#     fi_info += fi_info_new
#     # print('fi', fi_info)
#     # fi_matrix.symmetrie_test(fi_info_scale)

#     det_fi_scale = torch.det(fi_info_scale)
#     log_det_scale = torch.log(det_fi_scale)

#     det_fi = torch.det(fi_info)
#     log_det = torch.log(det_fi)
#     print(det_fi)

#     step_reward_scale = log_det_scale - log_det_previous_scale
#     # print('step reward', step_reward_scale.item())
#     total_reward_scale = total_reward_scale + step_reward_scale.item()
#     # print("reward", total_reward_scale)

#     scale_factor = (det_init / det_fi_scale).pow(1 / 2)
#     fi_info_previous_scale = fi_info_scale * (scale_factor / scale_factor_previous)
#     log_det_previous_scale = torch.slogdet(fi_info_previous_scale)[1]
#     scale_factor_previous = scale_factor
#     log_det_previous = log_det
#     # print(step_reward_scale)
#     # print(step_reward)
#     print('**************************************************************')
#     # time.sleep(1)

# # print("total_reward_scale", total_reward_scale.item())
# print("total_reward", total_reward)
# print('logdet fi', log_det)
# print('fi', fi_info)
