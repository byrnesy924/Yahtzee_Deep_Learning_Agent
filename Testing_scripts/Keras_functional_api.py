import numpy as np
import tensorflow as tf


import numpy as np
import tensorflow as tf

# Fake environment with 13 states and 4 possible actions
num_states = 13
num_actions = 4

# Generate random data for input
num_samples = 1000
states = np.random.rand(num_samples, num_states)
actions = np.random.randint(0, num_actions, size=num_samples)
rewards = np.random.rand(num_samples)

# Function to build the Q-learning network using TensorFlow
def build_q_learning_network(num_states, num_actions):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(num_states,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_actions)
    ])
    return model

# Build the Q-learning network
q_learning_model = build_q_learning_network(num_states, num_actions)

# Compile the model
q_learning_model.compile(optimizer='adam', loss='mse')

# Train the model
q_learning_model.fit(states, tf.keras.utils.to_categorical(actions, num_actions), epochs=100, verbose=1)


print(q_learning_model.summary())
print(q_learning_model.output_shape)
