import argparse
import os
import numpy as np
import torch

class FI_matrix(object):

    def __init__(self):
        # Define parameters
        self.J = 4.624e-06  # Moment of inertia
        self.km = 21.7e-03  # Motor constant
        self.gamma = 1.889e-05  # Proportional constant
        self.k1 = 23.04  # Elasticity constant
        self.fc = 10.37e-3  # Coulomb friction coefficient
        self.epsilon = 0.5  # Zero velocity bound [rad/s]
        self.fv = torch.tensor([2.16e-5], dtype=torch.float64, requires_grad=True)  # Viscous friction coefficient
        self.dt = 0.001

    def f(self, x, u, theta):
        # x1, x2 = x
        x1, x2 = x[0], x[1]
        fv = theta[0]
        km = self.km
        k1 = self.k1
        fc = self.fc
        # fv = theta
        dx1 = x2
        Tm = km * u
        Tl = self.gamma * k1 * torch.max(x1, torch.tensor(0.0, dtype=torch.float64))
        Tf = ((fc * torch.sign(x2) + fv * x2) / self.epsilon) * torch.min(torch.abs(x2), torch.tensor(self.epsilon, dtype=torch.float64))
        dx2 = (Tm - Tl - Tf) / self.J
        return torch.stack([dx1, dx2])

    def jacobian_f(self, x, u, theta):
        # 确保 x 启用了梯度计算
        # x = torch.tensor(x, requires_grad=True)
        x = x.clone().detach().requires_grad_(True)
        
        # 计算 f(x, u, theta)
        f_x = self.f(x, u, theta)

        # 创建一个列表来存储每个 f_x 分量相对于 x 的梯度
        jacobian = []
        
        # 对 f_x 的每个分量计算相对于 x 的梯度
        for i in range(f_x.size(0)):
            grad_x = torch.autograd.grad(f_x[i], x, retain_graph=True, create_graph=True)[0]
            jacobian.append(grad_x)
        
        # 将所有梯度组合为一个矩阵
        jacobian_matrix = torch.stack(jacobian)
        
        return jacobian_matrix

x_0 = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
u = 2.0
theta = torch.tensor([2.16e-5], dtype=torch.float64, requires_grad=True)

# 创建模型实例
fi_matrix = FI_matrix()

# 计算雅各比矩阵
jacobian_matrix = fi_matrix.jacobian_f(x_0, u, theta)

print(jacobian_matrix)

