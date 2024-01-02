import os
import random
from multiprocessing import Pool
from itertools import cycle

from Yahtzee import Yahtzee
from NNQmodel import NNQPlayer

"""
This is a rudimentary setup for using a random sampling approach to exploring the hyperparameter space
"""

def test_model(learning_rate, gamma, reward_for_all_dice, reward_factor_for_initial_dice_picked,
               reward_factor_for_picking_choice_correctly, name):
    """"""
    # TODO add in architecture
    model = NNQPlayer(
        show_figures=False,
        learning_rate=learning_rate,
        gamma=gamma,
        reward_for_all_dice=reward_for_all_dice,
        reward_factor_for_initial_dice_picked=reward_factor_for_initial_dice_picked,
        reward_factor_for_picking_choice_correctly=reward_factor_for_picking_choice_correctly,
        name=name
    )
    model.run(32, 16, save_results=False, save_model=False, verbose=False)  # Should take around 4 hours per item
    results = [learning_rate,
               gamma,
               reward_for_all_dice,
               reward_factor_for_initial_dice_picked,
               reward_factor_for_picking_choice_correctly,
               model.average_score,
               # model.average_loss
               ]
    return results


def randomly_sample_hyper_parameters(list_of_values, no_samples):
    return random.sample(list_of_values, no_samples)

if __name__ == '__main__':
    # TODO - try a manual method of bayesian optimisation instead

    # Define hyperparamters as discrete ranges
    learning_rate = [0.000_000_1 * i for i in range(1, 101)]
    gamma = [0.95 + i * 0.0005 for i in range(1, 101)]
    reward_for_all_dice = [0.5 * i for i in range(0, 21)]
    reward_factor_for_initial_dice_picked = [0.01 * i for i in range(0, 101)]
    reward_factor_for_picking_choice_correctly = [0.5 * i for i in range(0, 21)]
    name = list(range(1, 101))

    hyperparameter_space = [item for item in zip(learning_rate,
                                                 gamma,
                                                 cycle(reward_for_all_dice),
                                                 reward_factor_for_initial_dice_picked,
                                                 cycle(reward_factor_for_picking_choice_correctly),
                                                 name)
                            ]


    no_processes = os.cpu_count() - 2
    hyperparameters_to_test = randomly_sample_hyper_parameters(list_of_values=hyperparameter_space,
                                                               no_samples=no_processes)

    with Pool(no_processes) as pool:
        result = pool.starmap(test_model, hyperparameters_to_test)
    print(result)