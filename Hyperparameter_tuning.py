import os
import random
from multiprocessing import Pool
# from itertools import product  # Causes a Memory erro in this caser
from prodius import product  # Replace with open source solution https://github.com/sekgobela-kevin/prodius/

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
    model.run(2, 2, save_results=False, save_model=False, verbose=False)  # Should take around 4 hours per item
    results = [learning_rate,
               gamma,
               reward_for_all_dice,
               reward_factor_for_initial_dice_picked,
               reward_factor_for_picking_choice_correctly,
               model.average_score,
               # model.average_loss
               ]
    return results


def randomly_sample_hyper_parameters(iterator_of_lists, no_samples):
    """This function both randomly samples and creates a cartesian product of input lists
    The reason it does both is because the Cartesian Product step on the full lists was causing Memopry Errors on my
    machine
    """
    # Reduces Memory use which was causing me memory issues
    sampled_lists = [random.sample(item,no_samples*5) for item in iterator_of_lists]
    cartesian_product = list(product(*sampled_lists))
    return random.sample(cartesian_product, no_samples)

if __name__ == '__main__':
    # TODO - try a manual method of bayesian optimisation instead

    # Define hyperparamters as discrete ranges
    learning_rate = [0.000_000_1 * i for i in range(1, 101)]
    gamma = [0.95 + i * 0.0005 for i in range(1, 101)]
    reward_for_all_dice = [0.5 * i for i in range(0, 21)]
    reward_factor_for_initial_dice_picked = [0.01 * i for i in range(0, 101)]
    reward_factor_for_picking_choice_correctly = [0.5 * i for i in range(0, 21)]

    lists_to_product = [learning_rate, gamma, reward_for_all_dice,
                        reward_factor_for_initial_dice_picked, reward_factor_for_picking_choice_correctly]

    hyperparameter_space = product(*lists_to_product)  #Cartesian product of lists
    hyperparameter_space = list(hyperparameter_space) # split across two lines to reduce memory

    no_processes = os.cpu_count() - 2
    # hyperparameters_to_test = randomly_sample_hyper_parameters(list_of_values=hyperparameter_space,
    #                                                            no_samples=no_processes)
    # for hyperparameters in hyperparameters_to_test:
    #     hyperparameters.append("HParameter_testing_" + str(hyperparameters)) # This takes a string literal of the
    #     # hyperparameters
    #     # And tacks it onto the list as a name to be passed to the Yahtzee Model
    #     # It takes advantage of lists being mutable - careful with multiprocessing
    #
    # print(hyperparameters_to_test)
    # with Pool(no_processes) as pool:
    #     result = pool.starmap(test_model, hyperparameters_to_test)
    # print(result)