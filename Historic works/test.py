import casadi as ca
import numpy as np

# Define system parameters
J = 4.624e-06
gamma = 1.889e-05
epsilon = 0.5
Jm = 4.624e-06
N = 10  # Prediction horizon

# Define CasADi symbolic variables
x = ca.MX.sym('x', 2)           # State variables (2x1)
u = ca.MX.sym('u', N)           # Control inputs sequence (Nx1)
theta = ca.MX.sym('theta', 4)   # System parameters (4x1)
chi = ca.MX.sym('chi', 2, 4)    # Sensitivity matrix (2x4)
fi_info = ca.MX.sym('fi_info', 4, 4)  # Fisher Information Matrix (4x4)
dt = ca.MX.sym('dt')            # Time step (scalar)
R = ca.MX.sym('R')              # Noise variance (scalar)

# Define the system dynamics in CasADi
def f_casadi(x, u, theta):
    x1, x2 = x[0], x[1]
    km, k1, fc, fv = theta[0], theta[1], theta[2], theta[3]
    dx1 = x2
    Tm = km * u
    Tl = gamma * k1 * ca.fmax(x1, 0.0)
    Tf = ((fc * ca.sign(x2) + fv * x2) / epsilon) * ca.fmin(ca.fabs(x2), epsilon)
    dx2 = (Tm - Tl - Tf) / Jm
    return ca.vertcat(dx1, dx2)

# Define Jacobians
f_x = f_casadi(x, u, theta)
J_f_x = ca.jacobian(f_x, x)
J_f_theta = ca.jacobian(f_x, theta)

# Define the output equation
h = x[0]

# Jacobian of the output equation
J_h = ca.jacobian(h, x)

# Define the cost function
def compute_total_reward_casadi(u_seq, x_init, chi_init, fi_info_init, theta, other_parameters):
    dt, R = other_parameters
    x = x_init
    chi_local = chi_init
    fi_info_local = fi_info_init
    log_det_previous = ca.log(ca.det(fi_info_local))
    total_reward_local = 0
    
    for k in range(N):
        u_val = u_seq[k]
        f_x = f_casadi(x, u_val, theta)
        x = x + dt * f_x  # Update state symbolically
        
        J_f_x_val = J_f_x
        J_f_theta_val = J_f_theta
        
        # Ensure matrix dimensions are consistent
        chi_local = chi_local + dt * (ca.mtimes(J_f_x_val, chi_local) + J_f_theta_val)
        
        J_h_val = ca.MX([[1, 0]])  # Jacobian of the output equation
        dh_theta = ca.mtimes(J_h_val, chi_local)
        
        fi_info_new = ca.mtimes(dh_theta.T, dh_theta) * (1 / R)
        fi_info_local += fi_info_new
        
        det_fi = ca.det(fi_info_local)
        log_det = ca.log(det_fi)
        
        step_reward_scale = log_det - log_det_previous
        total_reward_local += step_reward_scale
        
        log_det_previous = log_det  # Update log_det_previous for the next iteration
    
    return -total_reward_local  # Negative because we use a minimizer

# Define symbolic optimization variables
u_sym = ca.MX.sym('u', N)
x0 = ca.MX.sym('x0', 2)

# Set up the optimization problem
cost = compute_total_reward_casadi(u_sym, x0, chi, fi_info, theta, [dt, R])
nlp = {'x': u_sym, 'f': cost}

# Set solver options
opts = {'ipopt': {'print_level': 0}, 'print_time': 0}

# Create the solver
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Example usage
u0 = np.zeros(N)  # Initial guess for the control input
sol = solver(x0=u0)

print("Optimal control sequence:", sol['x'])



# tensor1 = tf.constant([
#     [5.57754767e+13, -4.15624534e+08, -4.32369666e+14, -3.88712080e+11],
#     [-4.15624534e+08, 3.09712733e+03, 3.22190776e+09, 2.89658263e+06],
#     [-4.32369666e+14, 3.22190776e+09, 3.35171547e+15, 3.01328330e+12],
#     [-3.88712080e+11, 2.89658263e+06, 3.01328330e+12, 2.70902357e+09]
# ], shape=(4, 4), dtype=tf.float64)

# tensor2 = tf.constant([
#     [5.67550651e+13, -4.22924244e+08, -4.39963403e+14, -3.95539056e+11],
#     [-4.22924244e+08, 3.15152341e+03, 3.27849487e+09, 2.94745598e+06],
#     [-4.39963403e+14, 3.27849487e+09, 3.41058187e+15, 3.06620579e+12],
#     [-3.95539056e+11, 2.94745598e+06, 3.06620579e+12, 2.75660233e+09]
# ], shape=(4, 4), dtype=tf.float64)

# tensor3 = tf.constant([
#     [9.79588395e+11, -7.29971004e+06, -7.59373684e+12, -6.82697602e+09],
#     [-7.29971004e+06, 5.43960780e+01, 5.65871109e+07, 5.08733522e+04],
#     [-7.59373684e+12, 5.65871109e+07, 5.88663969e+13, 5.29224923e+10],
#     [-6.82697602e+09, 5.08733522e+04, 5.29224923e+10, 4.75787604e+07]
# ], shape=(4, 4), dtype=tf.float64)
# diag = np.eye(4) * 1e-6
# matrix = np.random.rand(4,4)
# print(np.linalg.eigvals(diag))
# print(np.linalg.eigvals(matrix))
# print(np.linalg.eigvals(diag+matrix))

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