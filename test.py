import numpy as np
import tensorflow as tf
import scipy

# a = np.array([[2.01597488e+08,  5.76509836e+06, -8.22060250e+09,  6.62295319e+06],
#               [5.76509836e+06,  3.26438031e+05, -2.44587869e+08,  2.62716393e+05],
#               [-8.22060250e+09, -2.44587869e+08,  3.35772947e+11, -2.74378666e+08],
#               [6.62295319e+06,  2.62716393e+05, -2.74378666e+08,  2.50853540e+05]])

# a = np.array([[ 5.67550651e+13, -4.22924244e+08, -4.39963403e+14, -3.95539056e+11],
#               [-4.22924244e+08,  3.15152341e+03,  3.27849487e+09,  2.94745598e+06],
#               [-4.39963403e+14,  3.27849487e+09,  3.41058187e+15,  3.06620579e+12],
#               [-3.95539056e+11,  2.94745598e+06,  3.06620579e+12,  2.75660233e+09]])

tensor1 = tf.constant([
    [5.57754767e+13, -4.15624534e+08, -4.32369666e+14, -3.88712080e+11],
    [-4.15624534e+08, 3.09712733e+03, 3.22190776e+09, 2.89658263e+06],
    [-4.32369666e+14, 3.22190776e+09, 3.35171547e+15, 3.01328330e+12],
    [-3.88712080e+11, 2.89658263e+06, 3.01328330e+12, 2.70902357e+09]
], shape=(4, 4), dtype=tf.float64)

tensor2 = tf.constant([
    [5.67550651e+13, -4.22924244e+08, -4.39963403e+14, -3.95539056e+11],
    [-4.22924244e+08, 3.15152341e+03, 3.27849487e+09, 2.94745598e+06],
    [-4.39963403e+14, 3.27849487e+09, 3.41058187e+15, 3.06620579e+12],
    [-3.95539056e+11, 2.94745598e+06, 3.06620579e+12, 2.75660233e+09]
], shape=(4, 4), dtype=tf.float64)

tensor3 = tf.constant([
    [9.79588395e+11, -7.29971004e+06, -7.59373684e+12, -6.82697602e+09],
    [-7.29971004e+06, 5.43960780e+01, 5.65871109e+07, 5.08733522e+04],
    [-7.59373684e+12, 5.65871109e+07, 5.88663969e+13, 5.29224923e+10],
    [-6.82697602e+09, 5.08733522e+04, 5.29224923e+10, 4.75787604e+07]
], shape=(4, 4), dtype=tf.float64)

print(np.linalg.det(tensor1))

# print(np.linalg.det(b))
# print(tf.linalg.det(b))

# def minor(matrix, i, j):
#     """
#     Returns the minor of the matrix excluding the i-th row and j-th column.
#     """
#     minor = np.delete(matrix, i, axis=0)
#     minor = np.delete(minor, j, axis=1)
#     return minor

# def determinant(matrix):
#     """
#     Recursively calculates the determinant of a matrix.
#     """
#     # Base case for 2x2 matrix
#     if matrix.shape == (2, 2):
#         return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
#     det = 0
#     for col in range(matrix.shape[1]):
#         det += ((-1) ** col) * matrix[0, col] * determinant(minor(matrix, 0, col))
#     return det

# # Example 4x4 matrix
# b = np.array([[1, 1e9],
#              [1e9, (1e18+100)]])

# print("Determinant:", determinant(a))