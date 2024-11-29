import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.normal import Normal
import math


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
        self.k1 = 23.04  # Elasticity constant
        self.fc = 10.37e-3  # Coulomb friction coefficient
        self.epsilon = 0.5  # Zero velocity bound [rad/s]
        self.fv = torch.tensor([2.16e-5], dtype=torch.float64)  # Viscous friction coefficient
        self.dt = 0.001
        self.R_inv = torch.tensor([[1e6, 0],[0, 1]], dtype=torch.float64) # Invers matrix of the sensor noise

    def f(self, x, u, theta):
        x1, x2 = x[0], x[1]
        fv = theta[0]
        km = self.km
        k1 = self.k1
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
# x_0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
# chi = torch.zeros((2, 1), dtype=torch.float64)
# # fi_info = torch.eye(1, dtype=torch.float64) * 1e-6
# fi_info = torch.zeros((1,1), dtype=torch.float64)
# det_T = 0.001  # Time step
# theta = torch.tensor([2.16e-5], dtype=torch.float64)

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

# x2_value = []
# x2_value_dashed = []

# # Start the simulation
# for k in range(300):  # 350 = 0.35s
#     # u = torch.tensor(1.5 + 1.5 * torch.math.sin(2*pi*k/100 - pi/2), dtype=torch.float64)
#     u = 6
#     dx = fi_matrix.f(x, u, theta)
#     x = x + det_T * dx
#     # x2_value.append(x[1].item())
#     if k<200:
#         x2_value_dashed.append(500)
#         x2_value.append(500)
#     else:
#         x2_value_dashed.append(0)
#         x2_value.append(-500)
#     # x2_value.append(500)
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
#     log_det_scale = torch.log(det_fi)

#     step_reward_scale = log_det_scale - log_det_previous_scale
#     # print(step_reward_scale)
#     total_reward_scale = total_reward_scale + step_reward_scale

#     scale_factor = (det_init / det_fi_scale).pow(1 / 4)
#     fi_info_previous_scale = fi_info_scale * (scale_factor / scale_factor_previous)
#     log_det_previous_scale = torch.slogdet(fi_info_previous_scale)[1]
#     scale_factor_previous = scale_factor
#     # print('**************************************************************')

# # print('det scale is', torch.det(fi_info_scale).item())
# # print('log det scale is', torch.log(torch.det(fi_info_scale)).item())
# # print("total_reward_scale", total_reward_scale.item())
# print('fi', fi_info)


# # Parameters of the mass-spring-damper system
# m = 4.624e-6  # Mass (kg)
# k = 23.04 * 1.889e-5  # Spring constant (N/m)
# c = 2.16e-5  # Damping coefficient (Ns/m)
# F_max = 6 * c  # Maximum constant input force (N)

# # Time settings
# t_end = 0.301  # Simulation time in seconds
# dt = 1e-3  # Time step
# t = np.arange(0, t_end, dt)  # Time vector

# # Initial conditions
# x0 = 0.0  # Initial displacement (m)
# v0 = 0.0  # Initial velocity (m/s)
# state = np.array([x0, v0])  # Initial state: [displacement, velocity]
# state_dashed = np.array([x0, v0])
# # System matrices
# A = np.array([[0, 1], [-k / m, -c / m]])  # System matrix
# B = np.array([[0], [1 / m]])  # Input matrix

# # Preallocate arrays for results
# displacement = [0.0]
# velocity = [0.0]
# FIM = [0.0]

# displacement_dashed = [0.0]
# velocity_dashed = [0.0]
# FIM_dashed = [0.0]
# # Simulation loop
# for x2 in x2_value:
#     dx = A @ state + B.flatten() * x2 * c  # Derivative of the state
#     state = state + dx * dt
#     displacement.append(state[0])
#     velocity.append(state[1])
#     FIM.append(100*state[0]**2 + state[1]**2/250000)

# for x2 in x2_value_dashed:
#     dx_dashed = A @ state_dashed + B.flatten() * x2 * c  # Derivative of the state
#     state_dashed = state_dashed + dx_dashed * dt
#     displacement_dashed.append(state_dashed[0])
#     velocity_dashed.append(state_dashed[1])
#     FIM_dashed.append(100*state_dashed[0]**2 + state_dashed[1]**2/250000)
    

# # Convert results to numpy arrays
# displacement = np.array(displacement)
# velocity = np.array(velocity)
# x2_value.append(500)
# x2_value_dashed.append(0)
# plt.figure(figsize=(10, 14))  # 增加图片长度，减少宽度

# plt.subplot(4, 1, 1)
# plt.plot(t, x2_value, color="blue", linewidth=1.5, label="Input 1")
# plt.plot(t, x2_value_dashed, linestyle="--", color="blue", linewidth=1.5, label="Input 2")
# plt.xlabel("Time (s)", fontsize=12)
# plt.ylabel("Motor Velocity (rad/s)", fontsize=12)
# plt.title("(a) System Input", fontsize=14)
# plt.grid(alpha=0.6)
# plt.legend(loc="upper right", fontsize=10)

# # 图2：位移灵敏度
# plt.subplot(4, 1, 2)
# plt.plot(t, displacement, color="green", linewidth=1.5, label="Input 1")
# plt.plot(t, displacement_dashed, linestyle="--", color="green", linewidth=1.5, label="Input 2")
# plt.xlabel("Time (s)", fontsize=12)
# plt.ylabel("State Variable a", fontsize=12)
# plt.title("(b) Sensitivity of the Motor Position to the Viscous Friction", fontsize=14)
# plt.grid(alpha=0.6)
# plt.legend(loc="upper left", fontsize=10)

# # 图3：速度灵敏度
# plt.subplot(4, 1, 3)
# plt.plot(t, velocity, label="Input 1", color="red", linewidth=1.5)
# plt.plot(t, velocity_dashed, linestyle="--", color="red", linewidth=1.5, label="Input 2")
# plt.xlabel("Time (s)", fontsize=12)
# plt.ylabel("State Variable b", fontsize=12)
# plt.title("(c) Sensitivity of the Motor Velocity to the Viscous Friction", fontsize=14)
# plt.grid(alpha=0.6)
# plt.legend(loc="upper right", fontsize=10)

# # 图4：FIM递增
# plt.subplot(4, 1, 4)
# plt.plot(t, FIM, label="Input 1", color="purple", linewidth=1.5)
# plt.plot(t, FIM_dashed, linestyle="--", color="purple", linewidth=1.5, label="Input 2")
# plt.xlabel("Time (s)", fontsize=12)
# plt.ylabel(r"$FIM_k$", fontsize=12)
# plt.title("(d) Increment of Fisher Information Matrix", fontsize=14)
# plt.grid(alpha=0.6)
# plt.legend(loc="upper left", fontsize=10)
# # 调整布局
# # plt.tight_layout(h_pad=2.0)
# plt.tight_layout()
# plt.savefig("plot.svg", format="svg")

