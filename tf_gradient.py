import numpy as np
import matplotlib.pyplot as plt

if 20!=0 and 20%20==0 and True:
    print('1')



pos_start = 69.4297
vel_start = -20.8071
pos_end = 0
vel_end = 0
acc_start = 0
acc_end = 0
t_start = 0.3
t_end = 0.5

A = np.array([
    [1, t_start, t_start**2, t_start**3, t_start**4, t_start**5],
    [0, 1, 2*t_start, 3*t_start**2, 4*t_start**3, 5*t_start**4],
    [0, 0, 2, 6*t_start, 12*t_start**2, 20*t_start**3],
    [1, t_end, t_end**2, t_end**3, t_end**4, t_end**5],
    [0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4],
    [0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3]
    ])

b = np.array([pos_start, vel_start, acc_start, pos_end, vel_end, acc_end])


# 求解多项式系数
coeff = np.linalg.solve(A, b)

# 定义五次多项式函数
def quintic_polynomial(t, coeff):
    return coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3 + coeff[4]*t**4 + coeff[5]*t**5

def quintic_polynomial_dt(t, coeff):
    return coeff[1] + 2*coeff[2]*t + 3* coeff[3]*t**2 + 4*coeff[4]*t**3 + 5*coeff[5]*t**4

# 时间范围
t_vals = np.linspace(0.3, 0.5, 200)

# 计算每个时间点对应的角度值
theta_vals = [quintic_polynomial(t, coeff) for t in t_vals]
print(len(theta_vals))
theta_dt = [quintic_polynomial_dt(t, coeff) for t in t_vals]

plt.figure(figsize=(10, 6))

# 绘制角度曲线
plt.subplot(2, 1, 1)
plt.plot(t_vals, theta_vals, label="Angle (rad)")
plt.title("Quintic Polynomial Interpolation")
plt.xlabel("Time (ms)")
plt.ylabel("Angle (rad)")
plt.grid(True)
plt.legend()

# 绘制速度曲线
plt.subplot(2, 1, 2)
plt.plot(t_vals, theta_dt, label="Velocity (rad/ms)", color='r')
plt.xlabel("Time (ms)")
plt.ylabel("Velocity (rad/ms)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()