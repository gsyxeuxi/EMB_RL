import numpy as np
import tensorflow as tf

x = tf.convert_to_tensor(np.array([[1,0]]))
y = tf.convert_to_tensor(np.array([[1,2,3],[4,5,6]]))
c = 3
z = tf.multiply(x,c)
print(z)
print(tf.matmul(z,z,True))