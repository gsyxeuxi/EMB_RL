import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(12)
for i in range (5):
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # self.state = np.array([-5.0, -430.0, -5.0, -430.0, 0.0, 0.0, 0.0], dtype=np.float64) #for test from other start point
    # self.state[0] = self.state[2] = random.uniform(-self.pos_reset_range_high, self.pos_reset_range_high)
    # self.state[1] = self.state[3] = random.uniform(-self.vel_reset_range_high, self.vel_reset_range_high)
    # self.state[0] = self.state[2] = 79.28
    # self.state[1] = self.state[3] = -183.91

    # if sample the fv
    state[5] = random.uniform(1, 3)
    seed = None
    random.seed(seed)
    print(state[5])



# pos_start = 62.3297
# vel_start = -424.8071
# pos_end = 0
# vel_end = 0
# acc_start = 0
# acc_end = 0
# t_start = 0.3
# t_end = 0.5

# A = np.array([
#     [1, t_start, t_start**2, t_start**3, t_start**4, t_start**5],
#     [0, 1, 2*t_start, 3*t_start**2, 4*t_start**3, 5*t_start**4],
#     [0, 0, 2, 6*t_start, 12*t_start**2, 20*t_start**3],
#     [1, t_end, t_end**2, t_end**3, t_end**4, t_end**5],
#     [0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4],
#     [0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3]
#     ])

# b = np.array([pos_start, vel_start, acc_start, pos_end, vel_end, acc_end])


# # 求解多项式系数
# coeff = np.linalg.solve(A, b)

# # 定义五次多项式函数
# def quintic_polynomial(t, coeff):
#     return coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3 + coeff[4]*t**4 + coeff[5]*t**5

# def quintic_polynomial_dt(t, coeff):
#     return coeff[1] + 2*coeff[2]*t + 3* coeff[3]*t**2 + 4*coeff[4]*t**3 + 5*coeff[5]*t**4

# # 时间范围
# t_vals = np.linspace(0.3, 0.5, 200)

# # 计算每个时间点对应的角度值
# theta_vals = [quintic_polynomial(t, coeff) for t in t_vals]
# print(len(theta_vals))
# theta_dt = [quintic_polynomial_dt(t, coeff) for t in t_vals]

# plt.figure(figsize=(10, 6))

# # 绘制角度曲线
# plt.subplot(2, 1, 1)
# plt.plot(t_vals, theta_vals, label="Angle (rad)")
# plt.title("Quintic Polynomial Interpolation")
# plt.xlabel("Time (ms)")
# plt.ylabel("Angle (rad)")
# plt.grid(True)
# plt.legend()

# # 绘制速度曲线
# plt.subplot(2, 1, 2)
# plt.plot(t_vals, theta_dt, label="Velocity (rad/ms)", color='r')
# plt.xlabel("Time (ms)")
# plt.ylabel("Velocity (rad/ms)")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()