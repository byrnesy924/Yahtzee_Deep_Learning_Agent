"""
This project contains a neural network agent that learns to play the game Yatzhee.

The original inspiration came from playing Yahtzee with my partner's family, and wondering if:
1. Yahtzee was a game where strategy choices/policy could be optimised (and how)
2. What ML methods would be good for optimising this strategy

Further, I had never worked with deep learning before this point, and so it made sense
    to take a reinforcement learning approach.

The approach is a Double Deep Q Learning Method. To break this down:
- Q learning is a type of reinforcement learning. It is a model-free approach.
- Deep refers to the use of a neural network to impliment the Q learning algorithm.
    - Using a nueral network is not necessary, but as per (Kang, Schroeder 2018) below is necessary
- Double refers to using two networks - a normal model and a target
    - This is a technique that reduces maximisation bias and can improve policy choice

This is also a heirarchical approach (Max Q learning). The agent is rewarded for doing things like choosing scores and
    choosing dice, but also rewarded for the score at the end of each turn and their overall score in the game
    i.e. Q(s, a) = V(s, a) + C(s,a) where V is a subtask (e.g. choosing dice) and C is a completion task
    This reduces the sparcity of rewards, improving performance, learning and the final policy choice

Included in this project were a few other pieces:
- Measuring the noise by training the agent multiple with the same hyper-parameters and examining the results
- Optimising the hyperparameters with Bayesian Optimisation
    - There is also a script that uses a random tuning approach to optimisation

Other resources and inspiration for this project:
Initially used to help write code:
    https://medium.com/@carsten.friedrich/part-4-neural-network-q-learning-a-tic-tac-toe-player-that-learns-kind-of-2090ca4798d
Great article on using a double q learning model:
    https://medium.com/p/b6bf911b6b2c
Great article on Double Q learning approach:
    https://davidrpugh.github.io/stochastic-expatriate-descent/pytorch/deep-reinforcement-learning/deep-q-networks/2020/04/11/double-dqn.html
Another Q Learning approach related to Yahtzee :
    https://www.yahtzeemanifesto.com/reinforcement-learning-yahtzee.pdf
    ^ Where methods differ is in the implimentation of the game and heirarchy. This project assumes reroll always,
    which is more computationally expensive, but the agent does not need to learn to choose to reroll
    so it is easier to impliment
I found this PhD student had a very similar approach as well:
    https://raw.githubusercontent.com/philvasseur/Yahtzee-DQN-Thesis/dcf2bfe15c3b8c0ff3256f02dd3c0aabdbcbc9bb/webpage/final_report.pdf
Further reading includes the research of James Glenn, who also has a lot of work in this area.

"""
import time
import pandas as pd
import cProfile
import pstats

from pympler import asizeof  # Accurate memory analysis


from Yahtzee import Yahtzee, random_player_game
from NNQmodel import NNQPlayer


def profile_code(model):
    # Diagnose / profile code
    profile = cProfile.Profile()
    do_profile = False  # Change in order to run the profiler

    if do_profile:
        def wrapper():
            model.run(8, 4)
            return
        profile.runcall(wrapper)
        ps = pstats.Stats(profile)
        ps.print_stats()
    return ps


def check_all_variables(yahtzee_player):
    print("Local variables: \n ", asizeof.asized(locals(), detail=1).format())
    print("Dict of the yahtzee player: \n", asizeof.asized(yahtzee_player.__dict__, detail=1).format())
    print("Dir of variables is of size: \n ", asizeof.asized(dir(yahtzee_player), detail=1).format())
    print("Size of global variables: \n ", asizeof.asized(globals(), detail=1).format())


def long_term_model_training_with_resource_limitations():
        """This is a wrapper for loading previously trained weights and to continue training. In other words, train in batches. it has some flaws as epsilon resets 
            upon re-enstatiation of the NNQ model. However, this was the quickest temporary solution to resource contraints on my machines (I would run out of memroy after around 7k epochs).
            It also allows me to check the change in weights and dig into the model with a bit more detail
        """
   
        # check_all_variables(yahtzee_player)
        for i in range(training_runs):
            # Define name
            start = time.perf_counter()
            model_name = f"64x64x64x64x64_architecture_training_step_{i+1}"

            # Create and train model, with loading weights
            yahtzee_player = NNQPlayer(show_figures=False, name=model_name, **model_hyperparameters)
            if i > 0:
                # TODO remove this so that it loads the Current Model
                yahtzee_player.load_model(load_as_training_model=True)
            
            yahtzee_player.run(1, 1, save_results=False, save_model=False, verbose=False)  # use to instantiate the weights
            print("Weights before training: ")
            start_weights = yahtzee_player.dqn_model.get_weights()
            print(start_weights)

            yahtzee_player.run(epochs, games_per_epoch, save_results=True, save_model=True, verbose=False)

            print("Weights after training: ")
            print(yahtzee_player.dqn_model.get_weights())

            # TODO - log this information        
            # print(asizeof.asized(yahtzee_player.recorded_rewards, detail=1).format())
            # print(asizeof.asized(yahtzee_player.score_tracker_special, detail=1).format())
            # print(asizeof.asized(yahtzee_player.score_tracker_singles, detail=1).format())

            print(f"Took {(time.perf_counter() - start)/3600} hrs to run {epochs} with {games_per_epoch} # games in training step {i+1}")

            if i != training_runs:
                # Delete the object to reduce memory usage - # TODO research how python will allocate memory when re-assigning variable to a new instance of the object
                yahtzee_player.save_model(save_as_current_model=True)
                del yahtzee_player
        


if __name__ == '__main__':
    start = time.perf_counter()
    random_player = Yahtzee(player_type="random")
    random_results = [random_player_game(random_player=random_player) for i in range(100)]
    df = pd.DataFrame(random_results)
    # df.to_csv("1000 Random games.csv")
    print(f"RANDOM: Took {time.perf_counter() - start}s to play")
    print(df.describe())

    # Attempt to reduce memory usage
    del random_player, df

    # Define hyperparameters of Yahtzee Model. These were rounded averages from HParameter testing:
    model_hyperparameters = {
        "learning_rate": 0.000_05,
        "gamma": 0.92,
        "model_architecture": [16, 16, 16],
        "reward_for_all_dice": 5,
        "reward_factor_for_initial_dice_picked": 0.45,
        "reward_factor_for_picking_choice_correctly": 5.2,
        "reward_factor_total_score": 1.7,
        "reward_factor_chosen_score": 3.5,
        "punish_factor_not_picking_dice": -0.3,
        "punish_amount_for_incorrect_score_choice": -3,
        "batch_size": 400,
        "buffer_size": 100,
        "length_of_memory": 12_000,
    }

    start = time.perf_counter()

    # Train Model
    epochs = 1_000  
    games_per_epoch = 64
    training_runs = 4
    model_name = "Test_bug_fixes_on_NNQ_results"

    yahtzee_player = NNQPlayer(show_figures=False, name=model_name, **model_hyperparameters)
    yahtzee_player.run(epochs, games_per_epoch, save_results=True, save_model=True, verbose=False)
    
    # check_all_variables(yahtzee_player)
  
