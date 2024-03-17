"""
Inspiration for this project:
https://medium.com/@carsten.friedrich/part-4-neural-network-q-learning-a-tic-tac-toe-player-that-learns-kind-of-2090ca4798d
Also: https://medium.com/p/b6bf911b6b2c <- great article on using a target model
https://www.yahtzeemanifesto.com/reinforcement-learning-yahtzee.pdf <- Another Q learning approach.
    ^ Where methods differ is in structure of game and heirarchy. Mine assumes reroll always,
    which is more computationally expensive, but the agent does not need to learn to choose to reroll
    so it is easier to impliment
Plus playing Yahtzee with my Partner's family :)
TODO change this DocString and put in ReadMe and put acknowledgments

Also see:
https://raw.githubusercontent.com/philvasseur/Yahtzee-DQN-Thesis/dcf2bfe15c3b8c0ff3256f02dd3c0aabdbcbc9bb/webpage/final_report.pdf
^ This is a PHD student who did basically the same thing

Also looks like James Glenn is the guy in this field - at Stanford or Yale or something

Notes on approach:
- Uses Double Deep Q Learning method
- Also uses a Heirarchy approach (MaxQ Q-learning). The agent is rewarded for doing things like choosing scores and
    choosing dice, but also rewarded for the score at the end of each turn and their overall score in the game
    i.e. Q(s, a) = V(s, a) + C(s,a ) where V is a subtask (e.g. choosing dice) and C is a completion task
    This reduces the sparcity of rewards, greatly improving performance and learning

Also add write up about noise measurement - noise relative to signal


"""
import time
import pandas as pd
import cProfile
import pstats

from Yahtzee import Yahtzee
from NNQmodel import NNQPlayer


def random_player_game(random_player: Yahtzee):
    random_player.roll_dice()
    for i in range(12):
        for y in range(3):
            if random_player.sub_turn == 1:
                random_player.roll_dice()
            random_player.turn(player_input=False, random_choice=True)
    score = random_player.calculate_score()
    random_player.reset_game()
    return score


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


if __name__ == '__main__':
    start = time.perf_counter()
    random_player = Yahtzee(player_type="random")
    random_results = [random_player_game(random_player=random_player) for i in range(100)]
    df = pd.DataFrame(random_results)
    df.to_csv("1000 Random games.csv")
    print(f"RANDOM: Took {time.perf_counter() - start}s to play")
    print(df.describe())

    # Attempt to reduce memory usage
    del random_player

    # Define hyperparameters of Yahtzee Model. These were rounded averages from HParameter testing:
    # TODO architecture of the model itself
    model_hyperparameters = {
        "learning_rate": 0.000_25,
        "gamma": 0.92,
        "reward_for_all_dice": 5,
        "reward_factor_for_initial_dice_picked": 0.45,
        "reward_factor_for_picking_choice_correctly": 5.2,
        "reward_factor_total_score": 1.7,
        "reward_factor_chosen_score": 3.5,
        "punish_factor_not_picking_dice": -0.3,
        "punish_amount_for_incorrect_score_choice": -3,
        "batch_size": 100,
        "buffer_size": 100,
        "length_of_memory": 4800,
        "name": "Testing_HP_bug_fixed"
    }

    yahtzee_player = NNQPlayer(show_figures=True, **model_hyperparameters)
    start = time.perf_counter()

    # Train Model
    epochs = 8192
    games_per_eopch = 64
    yahtzee_player.run(epochs, games_per_eopch, save_results=False, save_model=False, verbose=False)
    print(f"Took {(time.perf_counter() - start)/3600} hours to run {epochs*games_per_eopch} games in {epochs} epochs")
