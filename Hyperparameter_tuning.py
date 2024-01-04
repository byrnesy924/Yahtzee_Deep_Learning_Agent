import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from time import perf_counter
from multiprocessing import Pool
from itertools import product  # Memory Error comes from wrapping in list()
# from prodius import product  # Replace with open source solution if Memory Err
# https://github.com/sekgobela-kevin/prodius/

from NNQmodel import NNQPlayer

"""
This is a rudimentary setup for using a random sampling approach to exploring and optimising hyperparameters
"""
# TODO impliemnt a bayesian approach instead


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

    model.run(512, 64, save_results=False, save_model=False, verbose=False)
    results = [learning_rate,
               gamma,
               reward_for_all_dice,
               reward_factor_for_initial_dice_picked,
               reward_factor_for_picking_choice_correctly,
               model.average_score,
               # model.average_loss
               ]
    return results


def randomly_sample_hyper_parameters(iterator, no_samples):
    """Randomly samples list"""
    # Reduces Memory use which was causing me memory issues
    return list(random.sample(iterator, no_samples))


def plot_hyperparameter_space(results: pd.DataFrame):
    """Plot the results from hyperparameter testing to identify possible configurations to use

    :param results: the dataframe with cols of HyperParameters and the average score of the trained model
    :type results: pd.DataFrame
    """
    vars = ["LR", "Gamma", "R1", "R2", "R3"]
    pair_plot = sns.pairplot(results, vars=vars, hue="Avg_Score")

    pair_plot.fig.suptitle('Pair Plot of HyperParameters with Color by Avg_Score', y=1.02, fontsize=16)
    plt.show()
    plt.savefig("Hyperparameter testing results.png")  # TODO put this somewhere rather in .\
    plt.close()
    return


def plot_correlation_heatmap(results):
    heatmap = sns.heatmap(results.corr()["Avg_Score"], annot=True)
    heatmap.fig.suptitle('HeatMAp of HyperParameters correlation with Avg_Score', y=1.02, fontsize=16)
    plt.show()
    plt.savefig("Hyperparameter heatmap.png")  # TODO put this somewhere rather in .\
    plt.close()
    return


def test_hyperparameters(list_of_hyperparameters, no_processes):
    # Sample again to explore hyperparameter space
    hyperparameters_pre_format = randomly_sample_hyper_parameters(
        list_of_hyperparameters, no_processes
        )

    hyperparameters_to_test = []  # List comprehension would be gross
    for hyperparameters in hyperparameters_pre_format:
        set_of_parameters = list(hyperparameters)
        set_of_parameters.append("HParameter_testing_" + str(hyperparameters))
        hyperparameters_to_test.append(tuple(set_of_parameters))
        # This takes a string literal of the hyperparameters
        # And tacks it onto the tuple as a name to be passed to the Yahtzee Model

    start = perf_counter()
    with Pool(no_processes) as pool:
        result = pool.starmap(test_model, hyperparameters_to_test)

    results = pd.DataFrame(result, columns=["LR", "Gamma", "R1", "R2", "R3", "Avg_Score"])
    print(f"Took {(perf_counter() - start)/60} mins to do {no_processes} tests")
    return results


if __name__ == '__main__':
    # TODO - try a manual method of bayesian optimisation instead

    # Define hyperparamters as discrete ranges
    learning_rate = [0.000_000_1 * i for i in range(1, 101)]
    gamma = [0.95 + i * 0.0005 for i in range(1, 101)]
    reward_for_all_dice = [0.5 * i for i in range(0, 21)]
    reward_factor_for_initial_dice_picked = [0.01 * i for i in range(0, 101)]
    reward_factor_for_picking_choice_correctly = [0.5 * i for i in range(0, 21)]
    # batch_size  # TODO - impliment these also, careful of curse of dimensionality
    # buffer_size
    # memory

    # Sample once to reduce load when doing cartesian product
    lists_to_product = [randomly_sample_hyper_parameters(learning_rate, 16),
                        randomly_sample_hyper_parameters(gamma, 16),
                        randomly_sample_hyper_parameters(reward_for_all_dice, 4),
                        randomly_sample_hyper_parameters(reward_factor_for_initial_dice_picked, 16),
                        randomly_sample_hyper_parameters(reward_factor_for_picking_choice_correctly, 4)]

    hyperparameter_space = product(*lists_to_product)  # Cartesian product of lists
    hyperparameter_space = list(hyperparameter_space)  # split across two lines to reduce memory
    no_processes = os.cpu_count() - 4
    no_runs_of_testing = 10

    start = perf_counter()
    list_of_results = [test_hyperparameters(hyperparameter_space, no_processes) for i in range(no_runs_of_testing)]
    results = pd.concat(list_of_results)

    if not os.path.isdir(f"Results\\{datetime.today().strftime('%Y-%m-%d')}_HParameter_testing_results"):
        os.makedirs(f"Results\\{datetime.today().strftime('%Y-%m-%d')}_HParameter_testing_results")

    results.to_csv(
        f"Results\\{datetime.today().strftime('%Y-%m-%d')}_HParameter_testing_results\\HParameter_testing_results.csv"
        )

    # Print to screen how long it took and results
    print(results.sort_values("Avg_Score", ascending=False))
    print(f"Took {(perf_counter() - start)/60} mins to run {no_processes*no_runs_of_testing} by {128*64} games")

    plot_hyperparameter_space(results)
    plot_correlation_heatmap(results)
