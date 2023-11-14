# This stub is for speed testing some particular pieces of code, instead of having to AB test the code base
import numpy as np
import tensorflow as tf
from time import perf_counter
import random


def original_batch_update(array):
    mini_batch = random.sample(array, 64)
    states = [i[0] for i in mini_batch]
    actions = [i[1] for i in mini_batch]
    rewards = [i[2] for i in mini_batch]
    next_states = [i[3] for i in mini_batch]
    dones = [i[4] for i in mini_batch]

    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    return


def new_batch_update(array):
    mini_batch = np.array(random.sample(array, 64), dtype=object)
    states = mini_batch[:, 0]  # .tolist()
    actions = mini_batch[:, 1]  # .tolist()
    rewards = mini_batch[:, 2]  # .tolist()
    next_states = mini_batch[:, 3]  # .tolist()
    dones = mini_batch[:, 4]  # .tolist()

    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    return


def new_test_numpy_random_lists(array):
    mini_batch = np.array(random.sample(array, 64), dtype=object)
    states = mini_batch[:, 0].tolist()
    actions = mini_batch[:, 1].tolist()
    rewards = mini_batch[:, 2].tolist()
    next_states = mini_batch[:, 3].tolist()
    dones = mini_batch[:, 4].tolist()

    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    return


def test_batch_arrays_for_update():
    array = np.random.rand(1000, 5).tolist()

    start_1 = perf_counter()

    for i in range(1000):
        original_batch_update(array)
    print(f"Batch array took {(perf_counter() - start_1)}s")

    start_2 = perf_counter()
    for i in range(1000):
        new_batch_update(array)
    print(f"New Batch array took {(perf_counter() - start_2)}s")

    start_3 = perf_counter()
    for i in range(1000):
        new_test_numpy_random_lists(array)
    print(f"New Batch array took {(perf_counter() - start_3)}s")



