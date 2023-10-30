import pandas as pd
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import time

from collections import deque

from Yahtzee import Yahtzee


# Create a custom Q-learning network using TensorFlow's subclassing API
class QLearningModel(tf.keras.Model):
    def __init__(self, num_actions, num_samples, num_states):
        """

        :param num_actions: number of outputs of the model, i.e. decisions of the model
        :param num_samples: the number of samples to train the model on
        :param num_states: the number of inputs, i.e. the states of the game
        """
        super(QLearningModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')  # TODO experiment with changed arhcitecture
        self.output_layer = tf.keras.layers.Dense(num_actions)
        self.gradients = None
        self.num_actions = num_actions  # Store number of actions

        # Define the loss function
        self.loss_function = tf.keras.losses.MeanSquaredError()

        # Define the optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        """ All this code is deprecated - it works by randomly generating data"""
        # Empty - batch fill with random generation
        # self.states = np.empty(num_samples, num_states)  # empty matrix of num_samples rows and num_states columns
        # self.actions = np.empty(num_samples, num_states)  # Vector of length num_samples
        # self.rewards = np.empty(num_samples)  # Vector of num_samples in length

        # Generate the random states themselves
        # self.states = np.random.rand(num_samples, num_states)  # matrix of num_samples rows and num_states columns
        #
        # # Actions: bool for each dice (5 zeros or ones), and 1-13 for the actual pick
        # actions_dice = np.random.randint(0, 2, size=(num_samples, 5))  # Randomly choose 5 dice
        # actions_choice = np.random.randint(0, 14, size=(
        # num_samples, 1))  # Randomly choose between 0 and 12 options (each score)
        # self.actions = np.concatenate([actions_dice, actions_choice], axis=1)
        #
        # self.rewards = np.random.rand(num_samples)  # Vector of num_samples in length

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

    @tf.function
    def train_step(self, states, actions, rewards):
        """ Training step function"""
        with tf.GradientTape() as tape:
            q_values = self(states)
            # one_hot_actions = tf.one_hot(actions, num_actions) # only necessary when it was choosing 1 action
            actions_float = tf.cast(actions, tf.float32)  # necessary for multiplication
            predicted_q_values = tf.reduce_sum(q_values * actions_float, axis=1)
            loss = self.loss_function(rewards, predicted_q_values)

        self.gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(self.gradients, self.trainable_variables))
        return loss

    def train_model_epoch(self, num_epochs, num_samples, batch_size):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for i in range(0, num_samples, batch_size):
                batch_states = self.states[i:i + batch_size]
                batch_actions = self.actions[i:i + batch_size]
                batch_rewards = self.rewards[i:i + batch_size]
                loss = self.train_step(batch_states, batch_actions, batch_rewards)
                print(f"  Batch {i // batch_size + 1}/{num_samples // batch_size} - Loss: {loss:.4f}")


class NNQPlayer(Yahtzee):
    """
    Implements a Yahtzee player based on a Reinforcement Neural Network learning the Yahtzee Q function
    """

    def game_state_to_nn_input(self) -> np.ndarray:
        """
        Converts a Yahtzee game state to an input feature vector for the Neural Network. The input feature vector
        is a float array of size 20.
        First 2 bits are the current turn state (turn number and sub turn number)
        The next 5 bits are the current state of the dice
        The next 13 are the current state of the players scores
        :return: The feature vector representing the input Yahtzee game state.
        """
        # Get the state of the dice
        if self.sub_turn == 1:
            current_dice = self.first_roll
        elif self.sub_turn == 2:
            current_dice = self.second_roll
        else:
            current_dice = self.third_roll

        # Convert the dictionary of the dice state into a numpy array
        dice_state_array = np.array(list(current_dice.values()))

        current_dice_saved = [dice for dice in self.dice_saved]  # List that comes from Yahtzee class
        current_dice_saved.extend([0] * (5 - len(current_dice_saved)))  # extend dice saved until it is length 5

        # Turn state array
        game_state_array = np.array([self.turn_number, self.sub_turn] + current_dice_saved)

        # TODO experiment with setting these to negative numbers to show it has been chosen
        # get the state of the players choices
        scores = ["ones", "twos", "threes", "fours", "fives", "sixes", "three_of_a_kind", "four_of_a_kind",
                  "full_house", "small_straight", "large_straight", "yahtzee", "chance"]
        player_state_list = [self.__getattribute__(score) for score in scores]
        player_state_array = np.array(player_state_list)

        res = np.concatenate(
            [game_state_array, dice_state_array, player_state_array], dtype=object
        )
        return res

    def __init__(self):
        """
        Inherits from Yahtzee, stores
        """
        super(Yahtzee, self).__init__()

        # Hyper parameters
        self.learning_rate = 0.001  # TODO experiment with hyper parameters
        self.gamma = 0.99

        # TODO experiment  6 outputs vs full 18 output

        self.batch_size = 64
        self.action_size = 6
        self.state_size = 25
        self.buffer_size = 32
        # TODO - experiment with 32 or 16 - hyperparameter

        self.dqn_model = QLearningModel(num_states=self.state_size, num_actions=self.action_size, num_samples=1000)
        self.dqn_target = QLearningModel(num_states=self.state_size, num_actions=self.action_size, num_samples=1000)
        self.optimizers = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, )

        self.memory = deque(maxlen=2000)

        super().__init__()

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state, epsilon):
        """ Implements an epsilon method for choosing the action - if a random number is < epsilon (i.e. a small
            percentage of the time) choose something randomly in order to increase exploration
        """
        q_values = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))
        if np.random.rand() <= epsilon:
            action_dice = np.random.randint(0, 2, size=5)
            action_choice = np.random.randint(0, 13, size=1)
            action = np.concatenate([action_dice, action_choice])
        else:
            # action_dice = tf.math.sigmoid(q_values)[0:5]  # convert them to Binary - either choose the dice or don't
            # action_choice = tf.math.round(q_values)
            # action = tf.concat(tf.cast(action_dice, tf.float32), action_choice)
            action = tf.round(q_values)[0]  # [0] because for whatever reason this is a nested object

        return action, q_values

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """ Method updates the weights of the dqn in order to learn from rewards"""
        mini_batch = random.sample(self.memory, self.batch_size)

        states = [i[0] for i in mini_batch]
        actions = [i[1] for i in mini_batch]
        rewards = [i[2] for i in mini_batch]
        next_states = [i[3] for i in mini_batch]
        dones = [i[4] for i in mini_batch]

        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # calculate the target q values by running the next states through the target DQN
            target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))

            # Convert q values, ie. the DQN output into an action vector
            action_dice = tf.math.sigmoid(target_q[:, :-1])  # convert them to Binary - either choose the dice or dont
            action_choice = tf.math.round(target_q[:, -1:])

            next_action = tf.concat([action_dice, action_choice], 1)
            target_value = tf.reduce_sum(next_action * target_q, axis=1)

            target_value = (1 - dones) * self.gamma * target_value + rewards
            # Sudo code - if done, then reward is just reward (1-done). If not, then add on the target q* learning rate

            main_q = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            main_value = tf.reduce_sum(actions * main_q, axis=1)

            # hand coded mean squared error between the two functions - could also use tf function
            error = tf.square(main_value - target_value) * 0.5
            error = tf.reduce_mean(error)

            dqn_grads = tape.gradient(error, dqn_variable)
            self.optimizers.apply_gradients(zip(dqn_grads, dqn_variable))

        return error

    def calculate_action(self, action):
        """
        Helper function that wraps formatting the action

        :param action:
        :return:
        """
        # We filter out all illegal moves by setting the probability to -1. We don't change the q values
        # as we don't want the NN to waste any effort of learning different Q values for moves that are illegal
        # anyway.

        if isinstance(action, np.ndarray):
            action_picked = action[-1]
        else:
            action_picked = action[-1].numpy()

        # If not correct turn then make -1
        if self.sub_turn != 3 or action_picked > 12 or action_picked < -1:
            # Only pick a score if the sub turn is three other wise only pick dice
            # Also if the action picked is out of range then make it -1
            action_picked = -1

        # Dice that have already picked cannot be re-picked. Iterate through the current roll
        # Get current roll
        active_roll_mapper = {1: "first_roll", 2: "second_roll", 3: "third_roll"}
        current_roll = self.__getattribute__(active_roll_mapper[self.sub_turn])

        if isinstance(action, np.ndarray):
            action_dice = action[0:5]
        else:
            action_dice = action[0:5].numpy()  # Handle when action is a tensor

        if len(action_dice) > 5:
            raise Exception(f"Picked more than 5 dice. Picked: {action_dice} and action was {action}")

        # Handle if floats come out of the dice choice - this might be redundant code
        for index, item in enumerate(action_dice):
            if item > 0.9:
                action_dice[index] = 1
            else:
                action_dice[index] = 0

        # Bodge mapper to convert the words "one","two", etc. to indexes in the np array
        key_mapper = {"one": 0, "two": 1, "three": 2, "four": 3, "five": 4}
        for key, value in current_roll.items():
            if value == 0:
                indexer = key_mapper[key]
                action_dice[indexer] = -1

        # Our next move is the one with the highest probability after removing all illegal ones.
        # score move
        scores_to_choose = ["ones", "twos", "threes", "fours", "fives", "sixes", "three_of_a_kind", "four_of_a_kind",
                            "full_house", "small_straight", "large_straight", "yahtzee", "chance"]
        score_move = None
        if self.sub_turn == 3:
            if action_picked > 12 or action_picked < -1:
                raise Exception(f"choice is out of range! tried to pick {action_picked}")
            score_move = scores_to_choose[
                int(action_picked)]  # string of which one to pick - int because action is float

        # Dice move
        dice_move = {"one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
        for key, act in zip(dice_move, action_dice):
            dice_move[key] = act

        return score_move, dice_move, action_picked

    def move(self, score_move, dice_move, action_picked):
        """
        Helper function to move in the yahtzee game
        Interacts with the Yahtzee game and makes a move based on the probabilities returned
        This also calculates the reward received
        Note that this is heavily coupled with calculate action - it was originally one function
        """
        current_number_dice_saved = len(self.dice_saved)
        current_score = self.calculate_score()
        self.turn(player_input=False, choice_dice=dice_move, choice_score=score_move)

        reward = self.calculate_score() - current_score  # TODO experiment with gained vs total score

        # Current Implementation - if it picks a score and its the wrong sub turn then penalise it
        # 1 August 2023 - removed this
        # TODO experiment with negative reward - findings, did not perform as well
        # if action_picked < 0:
        #     reward -= 1
        # Current implementation - if it picks a score on the third try, reward it slightly
        if self.sub_turn == 1:
            # Note that sub turn is INCREMENTED after self.turn which is above, therefore check if sub turn is 1
            # If the network picked an action at the end, reward it
            if 0 < action_picked < 14:
                reward *= 2  # Double the reward for successfully choosing a score

            # Current Implementation - If it picked all its dice, reward it more. If it didn't, punish it
            if self.sub_turn == 1 and len(self.dice_saved) == 5:
                reward += 3
            # else:
            #     reward -= 1
        else:
            # If it is not the last sub turn, reward it very slightly for picking its dice
            # Map this to: 1 dice = 0.1, 2 dice = 0.2, 3 dice = 0.3, 4 dice = 0.2, 5 dice = 0.1
            # 1 August update - only reward for NEW dice picked. Aso reduce by 0.1
            # number_dice_picked_reward = ((abs(3 - len(self.dice_saved) - current_number_dice_saved) * -1) + 3) / 10
            # reward += number_dice_picked_reward

            # New implimentation - just reward it for picking dice
            reward += 0.1 * (len(self.dice_saved) - current_number_dice_saved)

        # print(f"The dice move was: {dice_move} and the score move was: {score_move}, and the reward was: {reward}")
        return reward

    def run(self, number_of_epochs, games_per_epoch, save_model=True, save_results=True, verbose = False):
        """

        :param verbose: (Bool) Whether to print the game and score to the command line
        :param save_results: (Bool) pass argument to save results in csv format
        :param save_model: (Bool) pass argument to save model in .keras format
        :param games_per_epoch: How many games to play between printing results to the screen
        :param number_of_epochs: The number of games to learn from
        :return:
        """
        # Define epochs
        losses = []
        final_scores = []
        scorecards = []

        for epoch in range(number_of_epochs):
            for game in range(games_per_epoch):
                epsilon = 1 / (games_per_epoch * 0.1 + 1)
                self.reset_game()
                self.roll_dice()  # For some reason the dice aren't rolled at the start of the first game

                score = 0
                for i in range(13):  # Number of turns per game
                    for j in range(3):  # Number of sub turns
                        state = self.game_state_to_nn_input()
                        action, q_values = self.get_action(state, epsilon)
                        score_move, dice_move, action_picked = self.calculate_action(action)
                        reward = self.move(score_move, dice_move, action_picked)
                        # moving directly changes the game state stored in this object
                        next_state = self.game_state_to_nn_input()

                        done = False
                        if i == 12 and j == 2:
                            done = True  # The game is done at this stage
                        self.append_sample(state, action, reward, next_state, done)

                        score += reward

                        loss = None
                        if len(self.memory) > self.batch_size:
                            loss = self.update()
                            if game % self.buffer_size == 0:
                                # This buffer size is the distance between the target model and the actual model
                                # Using the target model technique reduces the effect of correlation between
                                # Sequential experiences, in other words, improving stability
                                self.update_target()
                final_scores.append(self.calculate_score())
                scorecards.append(self.print_scores(verbose=False))
                losses.append(loss)
                if verbose:
                    print(f"Finished Game number {game}. Loss is currently {loss}. Score was {self.calculate_score()}")
            print(f"Epoch {epoch} finished")

            # Log variables
            if epoch % 3 == 2:
                if save_results:
                    # Form of logging - save to a csv
                    start_time = time.perf_counter()
                    pd.DataFrame(self.memory).to_csv(f"Epoch {epoch} memory.csv")
                    print(f"Took {time.perf_counter() - start_time} seconds to save the memory for epoch {epoch}")

        # Save the TF model
        if save_model:
            self.dqn_model.save_weights("Model\QNN_Yahtzee_weights.ckpt") #TODO create a system of variables and saves?

        # Plot (and save) the results of training
        scores = pd.DataFrame([final_scores, losses]).transpose()
        if save_results:
            scores.to_csv("Final scores.csv")
        scores["Rolling average"] = scores.iloc[:, 0].rolling(100).mean()
        scores["Rolling standard deviation"] = scores.iloc[:, 0].rolling(100).std()

        plt.figure(figsize=(20, 20))
        plt.plot(final_scores)
        plt.plot(scores["Rolling average"])
        plt.plot(scores["Rolling average"] - scores["Rolling standard deviation"])
        plt.plot(scores["Rolling average"] + scores["Rolling standard deviation"])
        plt.savefig("Final score.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(losses)
        plt.savefig("Loss.png")
        plt.show()
        plt.close()

        return


# For testing
if __name__ == '__main__':
    start = time.perf_counter()
    agent = NNQPlayer()
    agent.run(75, 64)

    # TODO Plot the number of straights, Yahtzee's etc that it gets over time

    print(f"Took {time.perf_counter() - start} seconds")
