import numpy as np
from casadi import *
import tensorflow as tf
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
import math

class FI_matrix(object):

    def __init__(self) -> None:
        # Define parameters
        self.J = 4.624e-06  # Moment of inertia
        self.km = 21.7e-03  # Motor constant
        self.gamma = 1.889e-05  # Proportional constant
        self.k1 = 23.04  # Elasticity constant
        self.fc = 10.37e-3  # Coulomb friction coefficient
        self.epsilon = 0.5  # Zero velocity bound [rad/s]
        self.fv = 2.16e-5  # Viscous friction coefficient
        # self.Ts = 1.0 * (self.fc + self.fv * self.epsilon) # Static friction torque
        self.dt = 0.001
        self.theta_tensor = tf.convert_to_tensor([self.km, self.k1, self.fc, self.fv], dtype=tf.float64)

    """
        Define the system dynamics of lumped parameter EMB model
        dx/dt = f(x, u, theta)
        state variable:
            x1: motor position
            x2: motor velocity
        input:
            u: motor current
        parameter:
            theta: km, k1, fc, fv, J, gamma, epsilon
    """
    @tf.function
    def f(self, x, u, theta):
        x1, x2 = tf.unstack(x)
        km, k1, fc, fv= tf.unstack(theta)
        dx1 = x2
        Tm = km * u
        Tl = self.gamma * k1 * tf.maximum(x1, 0.0)
        Tf = ((fc * tf.sign(x2) + fv * x2) / self.epsilon) * tf.minimum(tf.abs(x2), self.epsilon)
        dx2 = (Tm - Tl - Tf) / self.J
        return tf.convert_to_tensor([dx1, dx2], dtype=tf.float64)

    def h(self, x):
        """
        Define the output equation
        y = h(x)
        output:
            y: motor position
        """
        return x[0]

    def jacobian_h(self, x):
        """
        Define the Jacobian matrix of function h, J_h
        y = h(x)
        output: J_h
        """
        x1, x2 = x
        dh_dx1 = 1
        dh_dx2 = 0
        return tf.convert_to_tensor([[dh_dx1, dh_dx2]], dtype=tf.float64)
    
    @tf.function
    def jacobian(self, x, u):
        """
        Define the Jacobian matrix of function f, J_f,
        and the matrix of df_dtheta with each parameter
        dx/dt = f(x, u, theta)
        output: J_f, df/dtheta_norm = df/dtheta @ dtheta/dtheta_norm
        """
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x)
            tape.watch(self.theta_tensor)
            f_x = self.f(x, u, self.theta_tensor)
        jacobian_df_dx = tape.jacobian(f_x, x)
        jacobian_df_dtheta = tape.jacobian(f_x, self.theta_tensor, UnconnectedGradients.ZERO)
        jacobian_dtheta_dtheta_norm = tf.linalg.diag(np.array(self.theta_tensor))
        jacobian_df_dtheta_norm = tf.matmul(jacobian_df_dtheta, jacobian_dtheta_dtheta_norm)
        return jacobian_df_dx, jacobian_df_dtheta_norm

    def sensitivity_x(self, J_f, df_dtheta, chi):
        """
        Define the sensitivity dx/dtheta with recursive algorithm
        chi(k+1) = chi(k) + dt * (J_x * chi(k) + df_dtheta)
        output: chi(k+1)
        """
        chi = chi + self.dt * (tf.matmul(J_f, chi) + df_dtheta)
        return chi  

    def sensitivity_y(self, chi, J_h):
        """
        Define the sensitivity dy/dtheta
        dh_dtheta(k) = J_h * chi(k)
        output: dh_dtheta(k)
        """
        dh_dtheta = tf.matmul(J_h, chi)
        return dh_dtheta
    
    def fisher_info_matrix(self, dh_dtheta, R=0.05):
        """
        Define the fisher infomation matrix M
        M = dh_dtheta.T * (1/R) * dh_dtheta
        output: fi_info
        """
        return tf.matmul(dh_dtheta, dh_dtheta, True) * (1/R)

    def symmetrie_test(self, x):
        """
        Test if a matrix is symmetrie
        """
        x = np.array(x)
        y = (x + np.transpose(x)) / 2
        if np.array_equal(x, y) == False:
            print('not symmetrie matrix')

        try:
            np.linalg.cholesky(x)
            return True
        except np.linalg.LinAlgError:
            return False



# Simulation parameters
x_0 = tf.Variable([0.0, 0.0], dtype=tf.float64)
chi = tf.convert_to_tensor(np.zeros((2, 4)), dtype=tf.float64)
fi_info = tf.convert_to_tensor(np.eye(4) * 1e-6, dtype=tf.float64)
det_init = tf.linalg.det(fi_info)
fi_matrix = FI_matrix()
x = x_0
scale_factor = 1
scale_factor_previous = 1
log_det_previous_scale = tf.math.log(tf.linalg.det(fi_info))
total_reward_scale = log_det_previous_scale

N = 10  # Prediction horizon
theta = np.array([21.7e-03, 23.04, 10.37e-3, 2.16e-5]) # [km, k1, fc, fv]
gamma = 1.889e-05
epsilon = 0.5
Jm = 4.624e-06
dt = 0.001
R = 0.05

# Define the CasADi symbolic variables
x_sym = SX.sym('x', 2)          # State vector
u_sym = SX.sym('u', N)          # Control input sequence
theta_sym = SX.sym('theta', 4)  # Parameters
chi_sym = SX.sym('chi', 2, 4)   # Sensitivity matrix
fi_info_sym = SX.sym('fi_info', 4, 4)  # Fisher Information Matrix

# Define the system dynamics in CasADi
def f_casadi(x, u, theta):
    x1, x2 = x[0], x[1]
    km, k1, fc, fv = theta[0], theta[1], theta[2], theta[3]
    dx1 = x2
    Tm = km * u
    Tl = gamma * k1 * fmax(x1, 0.0)
    Tf = ((fc * sign(x2) + fv * x2) / epsilon) * fmin(fabs(x2), epsilon)
    dx2 = (Tm - Tl - Tf) / Jm
    return vertcat(dx1, dx2)

# Define the Jacobians
def jacobian_f(x, u, theta):
    f_x = f_casadi(x, u, theta)
    J_f_x = jacobian(f_x, x)
    J_f_theta = jacobian(f_x, theta)
    return J_f_x, J_f_theta

# Cost function
def compute_total_reward_casadi(u_seq, x_init, chi_init, fi_info_init, theta):
    total_reward = 0
    
    x = x_init
    chi_local = chi_init
    fi_info_local = fi_info_init
    
    log_det_previous = log(det(fi_info_local))
    total_reward_local = 0
    
    for k in range(N):
        u = u_seq[k]
        dx = f_casadi(x, u, theta)
        x = x + dt * f_casadi(x, u, theta) # there is problem with updating x, can't solve it..................
        
        J_f_x, J_f_theta = jacobian_f(x, u, theta)
        chi_local = chi_local + dt * (mtimes(J_f_x, chi_local) + J_f_theta)
        
        J_h = SX(np.array([[1,0]]))  # Jacobian of the output equation
        dh_theta = mtimes(J_h, chi_local)
        
        fi_info_new = mtimes(dh_theta.T, dh_theta) * (1/R)
        fi_info_local += fi_info_new
        
        det_fi = det(fi_info_local)
        log_det = log(det_fi)
        
        step_reward_scale = log_det - log_det_previous
        total_reward_local += step_reward_scale
        
        log_det_previous = log_det  # Update log_det_previous for the next iteration
    
    return -total_reward_local  # Negative because we use a minimizer

# Define the optimization problem
u_seq_sym = SX.sym('u_seq', N)
other_para_sym = SX.sym('other_para', 2)
J = compute_total_reward_casadi(u_seq_sym, x_sym, chi_sym, fi_info_sym, theta_sym)
nlp = {'x': u_seq_sym, 'f': J}
opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = nlpsol('solver', 'ipopt', nlp, opts)

# Run the MPC simulation
for k in range(350):  # 350 = 0.35s
    u_sequence_initial = np.ones(N) * 2  # Initial guess for u_sequence
    x_val = np.array(x)
    chi_val = np.array(chi)
    fi_info_val = np.array(fi_info)
    
    sol = solver(x0=u_sequence_initial, lbx=0, ubx=6)
    u_sequence_opt = sol['x'].full().flatten()
    u = u_sequence_opt[0]  # Apply the first control input
    
    dx = fi_matrix.f(x, u, fi_matrix.theta_tensor)
    x = x + fi_matrix.dt * dx
    jacobian_matrix = fi_matrix.jacobian(x, u)
    J_f = jacobian_matrix[0]
    J_h = fi_matrix.jacobian_h(x)
    df_theta = jacobian_matrix[1]
    chi = fi_matrix.sensitivity_x(J_f, df_theta, chi)
    dh_theta = fi_matrix.sensitivity_y(chi, J_h)
    fi_info_new = fi_matrix.fisher_info_matrix(dh_theta)
    fi_info_new_scale = fi_info_new * scale_factor
    fi_info += fi_info_new_scale
    fi_matrix.symmetrie_test(fi_info)
    det_fi_scale = tf.linalg.det(fi_info)
    log_det_scale = tf.math.log(det_fi_scale)
    step_reward_scale = log_det_scale - log_det_previous_scale
    total_reward_scale += step_reward_scale
    scale_factor = (det_init / det_fi_scale) ** (1/4)
    fi_info = (fi_info / scale_factor_previous) * scale_factor
    log_det_previous_scale = np.linalg.slogdet(fi_info)[1]
    scale_factor_previous = scale_factor

print('det sacle is', np.linalg.det(fi_info))
print('log det scale is', np.log(np.linalg.det(fi_info)))
print("total_reward_scale", total_reward_scale)
