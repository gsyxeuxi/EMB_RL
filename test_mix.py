import tensorflow as tf
import numpy as np
import math

import numpy as np
import tensorflow as tf

filename = f"fi_info_10_iter_300.npz"


with np.load(filename) as data:
    C = data['fi_info_10']
print(C)

# 定义对称正定矩阵
A = tf.constant([[ 6.03337828e+16,  2.14090753e+15, -2.48437473e+18,  2.16746228e+15],
                 [ 2.14090753e+15,  7.59687997e+13, -8.81565241e+16,  7.69110787e+13],
                 [-2.48437473e+18, -8.81565241e+16,  1.02299533e+20, -8.92499736e+16],
                 [ 2.16746228e+15,  7.69110787e+13, -8.92499736e+16,  7.78650453e+13]], dtype=np.float64)
print(A-C)
B = np.array([
    [3.75050393e+17,  1.33081550e+16, -1.54434997e+19,  1.34733708e+16],
    [1.33081550e+16,  4.72221848e+14, -5.47991663e+17,  4.78084306e+14],
    [-1.54434997e+19, -5.47991663e+17,  6.35919036e+20, -5.54794775e+17],
    [1.34733708e+16,  4.78084306e+14, -5.54794775e+17,  4.84019544e+14]
], dtype=np.float64)
# 计算行列式
det_A = np.linalg.det(C)
det_B = tf.linalg.eigvals(C)
print("行列式:", det_A,'===============', det_B)

# 计算逆矩阵
C_inv_np = np.linalg.inv(C)
print("NumPy逆矩阵:")


# 确认对称性
print("NumPy逆矩阵是否对称:", np.allclose(C, C.T))

# 验证正定性
x = np.array([1, 1, 1, 1], dtype=np.float64)
x_T_A_inv_x = np.dot(x.T, np.dot(C_inv_np, x))
print("x^T A_inv x:", x_T_A_inv_x)

# 使用TensorFlow计算逆矩阵
A_tf = tf.convert_to_tensor(B, dtype=tf.float64)
A_inv_tf = tf.linalg.inv(A_tf)
print("TensorFlow逆矩阵:")
print(A_inv_tf.numpy())

# 确认对称性
print("TensorFlow逆矩阵是否对称:", np.allclose(A.numpy(), C))
