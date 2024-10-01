import numpy as np
import torch

# print(np.random.normal(0, 0.001))
# print(np.random.normal(0, 1))
R_inv = torch.tensor([[1e6, 0],[0, 1]], dtype=torch.float64)
dh_dtheta = torch.tensor([[1], [100]], dtype=torch.float64)
def fisher_info_matrix(dh_dtheta):
    """
    Define the fisher infomation matrix M
    M = dh_dtheta.T * (1/R) * dh_dtheta
    output: fi_info
    """
    # return torch.matmul(dh_dtheta.t(), dh_dtheta) * (1/R)
    return torch.matmul(torch.matmul(dh_dtheta.t(), R_inv), dh_dtheta)

a = fisher_info_matrix(dh_dtheta)
print(a)