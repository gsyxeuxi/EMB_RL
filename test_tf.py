import tensorflow as tf
import numpy as np


J = 4.624e-06  # Moment of inertia
km = 21.7e-03  # Motor constant
gamma = 1.889e-05  # Proportional constant
epsilon = 0.5

@tf.function
def f(x, u, theta):
    x1, x2 = tf.unstack(x)
    dx1 = x1
    km, k1, fc, fv, Ts = tf.unstack(theta)
    def clamp_force_condition():
        def moving():
            return (km / J) * u - (gamma * k1 / J) * x1 - (1 / J) * (fc * tf.sign(x2) + fv * x2)
        
        def not_moving():
            return tf.cond(
                km * u - gamma * k1 * x1 > Ts,
                lambda: (km / J) * u - (gamma * k1 / J) * x1 - Ts / J,
                lambda: 0.0
            )
        return tf.cond(tf.abs(x2) > epsilon, moving, not_moving)
        
    def no_clamp_force_condition():
        def moving():
            return (km / J) * u - (1 / J) * (fc * tf.sign(x2) + fv * x2)
        
        def not_moving():
            return tf.cond(
                km * u > Ts,
                lambda: (km / J) * u - Ts / J,
                lambda: 0.0
            )
        return tf.cond(tf.abs(x2) > epsilon, moving, not_moving)
    dx2 = tf.cond(x1 > 0, clamp_force_condition, no_clamp_force_condition)
    return tf.convert_to_tensor([dx1, dx2], dtype=tf.float32)

theta = tf.constant([21.7e-03, 23.04, 10.37e-3, 2.16e-5, 1.2 * (10.37e-3 + 2.16e-5 * 0.5)], dtype=tf.float32)
x = tf.constant([2.0,4.0], dtype=tf.float32)
u = tf.constant(4.0, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    ys = f(x, u, theta)
    grad = tape.jacobian(ys, x)
print(grad)