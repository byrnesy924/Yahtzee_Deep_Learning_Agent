"""
Inspiration for this project: https://medium.com/@carsten.friedrich/part-4-neural-network-q-learning-a-tic-tac-toe-player-that-learns-kind-of-2090ca4798d
Also: https://medium.com/p/b6bf911b6b2c <- great article on using a target model

Plus playing Yahtzee with my Partner's family :)

"""
# TODO set up a proper venv and requirements.txt and some acknowledgements

import time
import tensorflow as tf
import pandas as pd

from Yahtzee import Yahtzee
from NNQmodel import NNQPlayer, QLearningModel


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


if __name__ == '__main__':
    start = time.perf_counter()
    random_player = Yahtzee(player_type="random")
    random_results = [random_player_game(random_player=random_player) for i in range(10000)]
    df = pd.DataFrame(random_results)
    df.to_csv("1000 Random games.csv")
    print(f"RANDOM: Took {time.perf_counter() - start}s to play")
    print(df.describe())

    yahtzee_player = NNQPlayer()
    start = time.perf_counter()
    yahtzee_player.run(800, 128, save_results=False, save_model=True, verbose = False)
    print(time.perf_counter() - start)
