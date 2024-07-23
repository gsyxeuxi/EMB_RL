import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
        self.Ts = 1.0 * (self.fc + self.fv * self.epsilon) # Static friction torque
        self.dt = 0.001
        self.theta_tensor = tf.convert_to_tensor([self.km, self.k1, self.fc, self.fv, self.Ts], dtype=tf.float64)

    def T_s(self, x2, theta):
        km, k1, fc, fv, Ts = tf.unstack(theta)

        Ts_1  = ((fc * tf.sign(x2) + fv * x2) / self.epsilon) * tf.minimum(tf.abs(x2), self.epsilon)
        Ts_2 = fc * tf.sign(x2) * tf.minimum(tf.abs(x2), self.epsilon) + fv * x2
        return Ts_1, Ts_2

# Instantiate the FI_matrix class
fi_matrix = FI_matrix()
theta = tf.constant([21.7e-03, 23.04, 10.37e-3, 2.16e-5, 1.2 * (10.37e-3 + 2.16e-5 * 0.5)], dtype=tf.float64)
# Generate x2 values
x2_values = np.linspace(-10, 10, 500)

# Convert x2_values to TensorFlow tensor
x2_tensor = tf.convert_to_tensor(x2_values, dtype=tf.float64)

# Calculate Ts_1 and Ts_2
Ts_1, Ts_2 = fi_matrix.T_s(x2_tensor, theta)

# Convert Ts_1 and Ts_2 to NumPy arrays for plotting
Ts_1_values = Ts_1.numpy()
Ts_2_values = Ts_2.numpy()

# Plot Ts_1 and Ts_2
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x2_values, Ts_1_values, label='Ts_1')
plt.xlabel('x2')
plt.ylabel('Ts_1')
plt.title('Ts_1 vs x2')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x2_values, Ts_2_values, label='Ts_2')
plt.xlabel('x2')
plt.ylabel('Ts_2')
plt.title('Ts_2 vs x2')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
