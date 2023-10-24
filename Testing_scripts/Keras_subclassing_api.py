import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

""" This is a python stub for experimenting with the Keras subclassing API. This was my first interaction with this
API.
"""

# Fake environment with 13 states and 4 possible actions
num_states = 18
num_actions = 6

# Generate random data for input
num_samples = 1000
states = np.random.rand(num_samples, num_states)  # matrix of num_samples rows and num_states columns
# actions = np.random.randint(0, num_actions, size=num_samples)  # Vector of length num_samples

# Coding multiple actions per step
actions_dice = np.random.randint(0, 2, size=(num_samples, 5))  # Randomly choose 5 dice
actions_choice = np.random.randint(0, 14, size=(num_samples, 1))  # Randomly choose between 0 and 12 options (each score)
actions = np.concatenate([actions_dice, actions_choice], axis=1)
# actions = np.random.randint(0, 2, size=(num_samples, 6))  # Randomly choose 6 actions
print(actions.shape)

rewards = np.random.rand(num_samples)  # Vector of num_samples in length


# Create a custom Q-learning network using TensorFlow's subclassing API
class QLearningModel(tf.keras.Model):
    def __init__(self, num_actions, num_states):
        super(QLearningModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(num_states, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# Instantiate the custom Q-learning model
q_learning_model = QLearningModel(num_actions, num_states=num_states)

# Define the loss function
loss_function = tf.keras.losses.MeanSquaredError()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()


# Training step function
@tf.function
def train_step(states, actions, rewards):
    with tf.GradientTape() as tape:
        q_values = q_learning_model(states)
        # one_hot_actions = tf.one_hot(actions, num_actions) # only necessary when it was choosing 1 action
        actions_float = tf.cast(actions, tf.float32)  # necessary for multiplication
        predicted_q_values = tf.reduce_sum(q_values * actions_float, axis=1)
        print(actions_float, q_values, predicted_q_values, rewards)
        loss = loss_function(rewards, predicted_q_values)

    gradients = tape.gradient(loss, q_learning_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_learning_model.trainable_variables))
    return loss


# Training loop
num_epochs = 100
batch_size = 32

losses = []

print(q_learning_model.get_weights())

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for i in range(0, num_samples, batch_size):
        batch_states = states[i:i + batch_size]
        batch_actions = actions[i:i + batch_size]
        batch_rewards = rewards[i:i + batch_size]
        loss = train_step(batch_states, batch_actions, batch_rewards)
        losses.append(loss)
        print(f"  Batch {i // batch_size + 1}/{num_samples // batch_size} - Loss: {loss:.4f}")

plt.plot(losses)
plt.show()

print("@@@@@@@@@@@@@@@@@@@@@")
print(q_learning_model.get_weights())