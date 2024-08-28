import tensorflow as tf

import numpy as np
observation = np.array([150 ,200, 400, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10], dtype=np.float64)
states = tf.Variable(tf.zeros((512, 1, 13)), dtype=tf.float32)
# states = tf.zeros((512, 1, 13))
print(states[0])
states[0].assign(observation)
# states[0].assign(observation[np.newaxis, :].astype(np.float32))
print(states[0])
# b_obs = tf.reshape(states, (-1,) + observation.shape)
b_obs = tf.reshape(states, -1)

print(b_obs.shape)

# observation_norm = (observation - np.mean(observation)) / (np.std(observation) + 1e-8)
# print(observation_norm)

