import os
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import time

from datetime import datetime
from collections import deque

from Yahtzee import Yahtzee


# Create a custom Q-learning network using TensorFlow's subclassing API
class QLearningModel(tf.keras.Model):
    def __init__(self, num_actions, num_samples, num_states, num_nodes_per_layer=32):  # TODO Experiment with architecture as a hyper parameter
        """

        :param num_actions: number of outputs of the model, i.e. decisions of the model
        :param num_samples: the number of samples to train the model on
        :param num_states: the number of inputs, i.e. the states of the game
        """
        super(QLearningModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(num_nodes_per_layer, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_nodes_per_layer, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_nodes_per_layer, activation='relu')  # TODO experiment with changed arhcitecture
        self.output_layer = tf.keras.layers.Dense(num_actions)  # Method is to loop through no. layers and change
        # Also need to change call method

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
        """ Training step function
        This is deprecated - the actual method is implimented in the NNQ object, and uses a target
        model and an actual model. This is "Double Deep Q learning" See here https://towardsdatascience.com/double-deep-q-networks-905dd8325412

        """
        with tf.GradientTape() as tape:
            q_values = self(states)
            # one_hot_actions = tf.one_hot(actions, num_actions) # only necessary when it was choosing 1 action
            actions_float = tf.cast(actions, tf.float32)  # necessary for multiplication
            predicted_q_values = tf.reduce_sum(q_values * actions_float, axis=1)
            loss = self.loss_function(rewards, predicted_q_values)

        self.gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(self.gradients, self.trainable_variables))
        return loss

    # This train method is deprecated
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

    def __init__(self, large_nnq_output=True,
                 learning_rate=0.000001,
                 gamma=0.99,
                 reward_for_all_dice=5,
                 punish_for_not_picking_dice=False,
                 reward_factor_for_initial_dice_picked=0.1,
                 reward_factor_for_picking_choice_correctly=2,
                 length_of_memory=2000,
                 batch_size=64,
                 buffer_size=32,
                 show_figures=False,
                 name="Basic"
                 ):
        """
        Inherits from Yahtzee, stores
        :param verbose_nnq_output: (Bool) Experiment with two architectures - this is the version with more outputs
        """
        super(NNQPlayer, self).__init__()  # Originally passed Yahtzee to super - should be NNQPlayer

        # Hyper parameters
        self.learning_rate = learning_rate
        # See stack overflow below - learning rate was quite high at 0.001 and lead to NaN output
        # https://stackoverflow.com/questions/39714374/nan-results-in-tensorflow-neural-network?rq=4
        self.gamma = gamma

        # Done experiment  6 outputs vs full 18 output - refactored for 18
        self.action_size = 18  # One for each dice and one for each selection
        self.large_action_size = large_nnq_output  # This is a bool - store it
        if not large_nnq_output:
            self.action_size = 6

        self.batch_size = batch_size  # For learning - how many steps are sampled
        self.state_size = 25
        self.buffer_size = buffer_size  # Number of games between Target update and normal update

        # Hyperparameters of Reward Structure
        self.reward_for_all_dice = reward_for_all_dice  # the amount it gets for picking a dice
        self.punish_for_not_picking_dice = punish_for_not_picking_dice  # If it doesnt pick dice punish it
        self.reward_factor_for_initial_dice_picked = reward_factor_for_initial_dice_picked  # reward for each inital dice chosen (not at end of round)
        self.reward_factor_for_picking_choice_correctly = reward_factor_for_picking_choice_correctly  # Reward it for actually choosing something

        self.dqn_model = QLearningModel(num_states=self.state_size, num_actions=self.action_size, num_samples=1000)
        self.dqn_target = QLearningModel(num_states=self.state_size, num_actions=self.action_size, num_samples=1000)
        self.optimizers = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, )

        # Memory size to train on
        self.memory = deque(maxlen=length_of_memory)

        # Added these dataframes to track scores and plot them over time
        self.score_tracker_special = pd.DataFrame(
            columns=["3OAK", "4OAK", "LSTR", "HSTR", "FH", "YTZ"])
        self.score_tracker_singles = pd.DataFrame(
            columns=range(1, 7)
        )

        # Locations to save memory and results
        self.memory_path = "Memory\\" + datetime.today().strftime('%Y-%m-%d') + "_" + name
        self.results_path = "Results\\" + datetime.today().strftime('%Y-%m-%d') + "_" + name
        self.show_figures = show_figures

        if not os.path.isdir(self.memory_path):
            os.makedirs(self.memory_path)
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)

        # Overall score of the model - used for hyperparameter tuning
        self.average_score = None  # update after running
        self.average_loss = None  # update after running

        super().__init__()  # TODO this might be not necessary

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

    def update_target(self):
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state, epsilon):
        """ Implements an epsilon method for choosing the action - if a random number is < epsilon (i.e. a small
            percentage of the time) choose something randomly in order to increase exploration
        """
        # call method depending on what kind of output is used
        if self.large_action_size:
            action, q_values = self.get_action_large_output(state)
        else:
            action, q_values = self.get_action_small_output(state)

        if np.random.rand() <= epsilon:
            action_dice = np.random.randint(0, 2, size=5)
            action_choice = np.random.randint(0, 13, size=1)
            action = np.concatenate([action_dice, action_choice])

        return action, q_values

    def get_action_small_output(self, state):
        """ Method that gets the action and q values when the output layer is size 6
        :param state:
        :return:
        """
        q_values = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))
        action = tf.round(q_values)[0]  # [0] to get the list out of the numpy array - hack for downstream in code

        return action, q_values

    def get_action_large_output(self, state):
        """ Originally created all code with the small NNQ model output (6 nodes). This converts the large nnq model
            model output into that same format
        """
        q_values = self.dqn_model(tf.convert_to_tensor([state], dtype=tf.float32))

        if True is np.isnan(q_values):
            print("Found NaNs in the q values! They are:\n")
            print(q_values)
            print("The state passed in: \n")
            print(state)
        state_choice = tf.math.argmax(q_values[0][-13:])  # Take the choice as the highest val in last 13 spots

        state_choice = tf.cast(state_choice, tf.float32)
        action = tf.concat([q_values[0][0:5], [state_choice]], axis=0)
        return action, q_values

    def append_sample(self, state, action, reward, next_state, done, q_values):
        self.memory.append((state, action, reward, next_state, done, q_values))

    def update(self):
        """ Method updates the weights of the dqn in order to learn from rewards"""
        # Use this numpy array and to list - previously used list comprehension for every one
        # Which was very slow
        # memory is a deque object which has a max length of 2000
        mini_batch = np.array(random.sample(self.memory, self.batch_size), dtype=object)
        states = mini_batch[:, 0]
        actions = mini_batch[:, 1].tolist()  # This needs to list as it is an array of tensors AND arrays - TF dislikes
        rewards = mini_batch[:, 2]
        next_states = mini_batch[:, 3]
        dones = mini_batch[:, 4]

        # mini_batch = random.sample(self.memory, self.batch_size)
        # states = [i[0] for i in mini_batch]
        # actions = [i[1] for i in mini_batch]
        # rewards = [i[2] for i in mini_batch]
        # next_states = [i[3] for i in mini_batch]
        # dones = [i[4] for i in mini_batch]

        # Run the update function
        # Moved this off into another method to use the tf.function decorator and improve performance
        dqn_variable = self.dqn_model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # calculate the target q values by running the next states through the target DQN
            target_q = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))
            if self.large_action_size:
                # This is the updated method four output size of 18
                max_values = tf.reduce_max(target_q[:, -13:], axis=1)
                mask = tf.cast(tf.equal(target_q[:, -13:], tf.expand_dims(max_values, axis=1)), dtype=tf.float32)
                # mask is a matrix where the max Q value is a 1, and 0 all else
                next_action = tf.concat([target_q[:, 0:5], mask], axis=1)

                # Also have to one-hot encode actions. This is because they are stored
                # As integers in a column, and this needs to be expanded
                one_hot_actions = tf.one_hot(tf.transpose(tf.cast(actions[:, -1:], tf.int8)), depth=13, )
                actions = tf.concat([actions[:, 0:5], one_hot_actions[0]], axis=1)

            else:
                # Convert q values, ie. the DQN output into an action vector
                action_dice = tf.math.sigmoid(
                    target_q[:, :-1])  # convert them to Binary - either choose the dice or dont
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
            action_dice = action[0:5]
        else:
            action_picked = action[-1].numpy()
            action_dice = action[0:5].numpy()

            # If not correct turn then make -1
        if self.sub_turn != 3 or -1 > action_picked > 12:
            # Only pick a score if the sub turn is three other wise only pick dice
            # Also if the action picked is out of range then make it -1
            action_picked = -1

        # Dice that have already picked cannot be re-picked. Iterate through the current roll
        # Get current roll
        active_roll_mapper = {1: "first_roll", 2: "second_roll", 3: "third_roll"}
        current_roll = self.__getattribute__(active_roll_mapper[self.sub_turn])

        if len(action_dice) > 5:
            raise Exception(f"Picked more than 5 dice. Picked: {action_dice} and action was {action}")

        # Handle if floats come out of the dice choice - this might be redundant code
        action_dice = np.where(action_dice > 0.9, 1, 0)

        # For each dice in the current roll, if it is 0 (i.e. already chosen) then make -1
        for index, value in enumerate(current_roll.values()):
            if value == 0:
                action_dice[index] = -1

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
        dice_move_names = ["one", "two", "three", "four", "five"]
        dice_move = dict(zip(dice_move_names, action_dice))

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
                reward *= self.reward_factor_for_picking_choice_correctly  # Double the reward for successfully choosing a score

            # Current Implementation - If it picked all its dice, reward it more. If it didn't, punish it
            if self.sub_turn == 1 and len(self.dice_saved) == 5:
                reward += 3  # self.reward_for_all_dice
            elif self.punish_for_not_picking_dice and current_number_dice_saved > 0:
                reward -= 1
        else:
            # If it is not the last sub turn, reward it very slightly for picking its dice
            # Map this to: 1 dice = 0.1, 2 dice = 0.2, 3 dice = 0.3, 4 dice = 0.2, 5 dice = 0.1
            # 1 August update - only reward for NEW dice picked. Aso reduce by 0.1
            # number_dice_picked_reward = ((abs(3 - len(self.dice_saved) - current_number_dice_saved) * -1) + 3) / 10
            # reward += number_dice_picked_reward

            # New implimentation - just reward it for picking dice
            reward += self.reward_factor_for_initial_dice_picked * (len(self.dice_saved) - current_number_dice_saved)

        # print(f"The dice move was: {dice_move} and the score move was: {score_move}, and the reward was: {reward}")
        return reward

    def run(self, number_of_epochs, games_per_epoch, save_model=True, save_results=True, verbose=False):
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

                        # December 27 - Identifying why NaN appear and NNQ stops playing
                        # Reason - the learning value was too high and the model was diverging
                        if True in np.isnan(action):
                            print("Found a NaN!\n", "Epoch is: ", epoch, "\nGame is: ", game)
                            print("state, action, q values:")
                            print(state)
                            print(action)
                            print(q_values)

                        done = False
                        if i == 12 and j == 2:
                            done = True  # The game is done at this stage
                        self.append_sample(state, action, reward, next_state, done, q_values)

                        score += reward

                loss = None
                if len(self.memory) > self.batch_size:  # In the beginning cannot run update unl there are enough games
                    loss = self.update()
                    if game % self.buffer_size == 0:
                        # This buffer size is the distance between the target model and the actual model
                        # Using the target model technique reduces the effect of correlation between
                        # Sequential experiences, in other words, improving stability
                        self.update_target()
                final_scores.append(self.calculate_score())
                scorecards.append(self.print_scores(verbose=False))  # TODO check this verbose flag is passed correctly
                self.count_scores_to_plot_over_time()  # Appends relevant scores to a DataFrame for plting
                losses.append(loss)
                if verbose:
                    print(f"Finished Game number {game}. Loss is currently {loss}. Score was {self.calculate_score()}")

            # Log variables
            if epoch % 16 == 2:  # updated to 16 to save disk space of memory
                # print(f"Epoch {epoch} finished")  # Convenient to check on training progress
                if save_results:
                    # Form of logging - save to a csv
                    # start_time = time.perf_counter()  # Removed timing - know it takes ~1sec, this reduces print calls
                    pd.DataFrame(self.memory).iloc[:, 0:5].to_csv(f"{self.memory_path}\\Epoch {epoch} memory.csv")
                    # print(f"Took {time.perf_counter() - start_time} seconds to save the memory for epoch {epoch}")


        # Save the TF model and its results
        if save_model:
            self.dqn_model.save_weights(
                f"{self.results_path}\\\QNN_Yahtzee_weights.ckpt")

        # Plot (and save) the results of training
        scores = pd.DataFrame([final_scores, scorecards, losses]) \
            .transpose() \
            .rename({0: "Scores", 1: "Card", 2: "Loss"}, axis=1)

        self.average_score = sum(final_scores) / len(final_scores)  # Get the average score of the model
        # self.average_loss = sum(losses) / len(losses)
        if save_results:
            scores.to_csv(f"{self.results_path}\\Final scores.csv")
        scores["Rolling average"] = scores.iloc[:, 0].rolling(100).mean()
        scores["Rolling standard deviation"] = scores.iloc[:, 0].rolling(100).std()

        self.plot_games_over_time(scores=scores, losses=losses)  # Coupled with code above
        self.plot_scores_over_time()

        return

    def plot_games_over_time(self, scores, losses):
        """
        Method that plots game scores over time + average +- 1 standard deviation.
        :param scores: (pd.DataFrame) Coupled with scores from run method - calculated there
        :param losses: (pd.DataFrame) Coupled with losses from run method - calculated there
        :return: None
        """
        plt.figure(figsize=(20, 20))
        plt.plot(scores["Scores"])
        plt.plot(scores["Rolling average"])
        plt.plot(scores["Rolling average"] - scores["Rolling standard deviation"])
        plt.plot(scores["Rolling average"] + scores["Rolling standard deviation"])
        plt.savefig(f"{self.results_path}\\Final score.png")
        if self.show_figures:
            plt.show()
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(losses)
        plt.savefig(f"{self.results_path}\\Loss.png")
        if self.show_figures:
            plt.show()
        plt.close()

        return

    def count_scores_to_plot_over_time(self):
        """This method records how many and how effective the scoring is over time"""
        list_of_special = ["three_of_a_kind", "four_of_a_kind", "full_house", "small_straight", "large_straight",
                           "yahtzee"]
        special_scores = [1 if self.__getattribute__(item) > 0 else 0 for item in list_of_special]

        df_special_score = pd.DataFrame(special_scores).transpose()
        df_special_score.columns = self.score_tracker_special.columns

        single_scores_names = ["ones", "twos", "threes", "fours", "fives", "sixes"]
        single_scores = [self.__getattribute__(score) for score in single_scores_names]
        df_single_scores = pd.DataFrame(single_scores).transpose()
        df_single_scores.columns = self.score_tracker_singles.columns

        self.score_tracker_special = pd.concat(
            [self.score_tracker_special, df_special_score])  # unsure why pycharm cant see
        self.score_tracker_singles = pd.concat([self.score_tracker_singles, df_single_scores])  # these are DataFrames

    def plot_scores_over_time(self):
        plot_special_scores = self.score_tracker_special.cumsum()
        plot_special_scores.reset_index(inplace=True, drop=True)
        plot_special_scores.fillna(0, inplace=True)
        plot_special_scores.plot(title="Special scores over time")
        plt.savefig(f"{self.results_path}\\Special scores over time.png")
        if self.show_figures:
            plt.show()
        plt.close()

        # Plot a rolling average of these scores as they are discrete
        self.score_tracker_singles.reset_index(inplace=True, drop=True)  # Need to reset index as not concated properly
        self.score_tracker_singles.fillna(0, inplace=True)
        for col in self.score_tracker_singles:
            plt.figure(figsize=(20, 20))
            plt.plot(self.score_tracker_singles[col])  # Create a df with rolling avg and std dev
            rolling_df = pd.concat([self.score_tracker_singles[col].rolling(100).mean(),
                                    self.score_tracker_singles[col].rolling(100).std()
                                    ], axis=1)

            rolling_df.columns = ["Mean", "StdDev"]
            plt.plot(rolling_df["Mean"])  # Plot the rolling average
            plt.plot(rolling_df["Mean"] + rolling_df["StdDev"])  # Plot 1 std dev above
            plt.plot(rolling_df["Mean"] - rolling_df["StdDev"])  # Plot 1 std dev below
            plt.savefig(f"{self.results_path}\\{col+1}'s over time.png")
            if self.show_figures:
                plt.show()
            plt.close()


# For testing
if __name__ == '__main__':
    start = time.perf_counter()
    agent = NNQPlayer()
    agent.run(75, 64)

    print(f"Took {time.perf_counter() - start} seconds")
