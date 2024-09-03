import tensorflow as tf

import numpy as np
# observation = np.array([150 ,200, 400, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10], dtype=np.float64)
# states = tf.Variable(tf.zeros((512, 1, 13)), dtype=tf.float32)
# # states = tf.zeros((512, 1, 13))
# print(states[0])
# states[0].assign(observation)
# # states[0].assign(observation[np.newaxis, :].astype(np.float32))
# print(states[0])
# # b_obs = tf.reshape(states, (-1,) + observation.shape)
# b_obs = tf.reshape(states, -1)

# print(b_obs.shape)

# observation_norm = (observation - np.mean(observation)) / (np.std(observation) + 1e-8)
# print(observation_norm)
def get_model():
    # Create a simple model.
    inputs = tf.keras.Input(shape=(32,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")
    return model


model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
model.save("my_model.keras")

# It can be used to reconstruct the model identically.
reconstructed_model = tf.keras.models.load_model("my_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)