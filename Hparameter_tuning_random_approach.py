import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from time import perf_counter
from multiprocessing import Pool
from itertools import product  # Memory Error comes from wrapping in list()
# from prodius import product  # Replace with open source solution if Memory Err - slower solution
# https://github.com/sekgobela-kevin/prodius/

from NNQmodel import NNQPlayer

"""
This is a rudimentary setup for using a random sampling approach to exploring and optimising hyperparameters

Initial findings: unsurprisingly, a higher learning rate generated a higher average score in the limited testing range
    This was expected, as because of computational restraints the models weren't trained for very long
    Reward 3 (the reward for correctly picking a choice) was slightly negatively correlated with score
    gamma had a slight posotive correlation with average score
    R1 had very little correlation
    R2 had a slight negative correlation

    Rather than treating these hyperparameters as directly correlated, it was assumed that the linear relationship may be more complex
    Therefore, a Bayesian approach was also explored
"""

def test_model(epochs, learning_rate, gamma, reward_for_all_dice, reward_factor_for_initial_dice_picked,
               reward_factor_total_score, reward_factor_chosen_score,
               reward_factor_for_picking_choice_correctly, punish_amount_for_incorrect_score_choice,
               punish_for_not_picking_dice,
               batch_size, memory, buffer_size,
               name="HP_testing", random_results=False):
    """ this is a black box wrapper function that trains the Q-Learning Network
    """
    # TODO add in architecture
    model = NNQPlayer(
        show_figures=False,
        learning_rate=learning_rate,
        gamma=gamma,
        reward_for_all_dice=reward_for_all_dice,
        reward_factor_total_score=reward_factor_total_score,
        reward_factor_chosen_score=reward_factor_chosen_score,
        reward_factor_for_initial_dice_picked=reward_factor_for_initial_dice_picked,
        reward_factor_for_picking_choice_correctly=reward_factor_for_picking_choice_correctly,
        punish_amount_for_incorrect_score_choice=punish_amount_for_incorrect_score_choice,
        punish_factor_not_picking_dice=punish_for_not_picking_dice,
        batch_size=int(batch_size),  # Note these need to be integers - coerce them to integers here for Bayesian Opt
        length_of_memory=int(memory),
        buffer_size=int(buffer_size),
        name=name
    )

    model.run(epochs, 64, save_results=False, save_model=False, verbose=False)
    results = [learning_rate,
               gamma,
               reward_for_all_dice,
               reward_factor_for_initial_dice_picked,
               reward_factor_for_picking_choice_correctly,
               batch_size,
               memory,
               buffer_size,
               model.average_score,
               model.average_loss
               ]

    # return results
    # In this random script, we return a list of results for formatting into a DataFrame
    # But in the Bayesian approach, we can only return a target variable, not a list
    if random_results:
        return results
    return model.average_score


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
    plt.savefig("Results\\Hyperparameter_testing\\Random_Hyperparameter_testing_results.jpeg") 
    plt.close()
    return


def plot_correlation_heatmap(results: pd.DataFrame()):
    heatmap = sns.heatmap(results.corr(), annot=True)
    # heatmap.fig.suptitle('HeatMAp of HyperParameters correlation with Avg_Score', y=1.02, fontsize=16)
    plt.show()
    plt.savefig("Results\\Hyperparameter_testing\\Random_Hyperparameter_heatmap.jpeg")
    plt.close()
    return heatmap


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
    learning_rate = [0.000_000_1 * i for i in range(1, 101)]  # TODO use np for these lines rather than list comp
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
    list_of_results = []
    for i in range(no_runs_of_testing):
        # Originally used list comprehension but ran into memory error
        results = test_hyperparameters(hyperparameter_space, no_processes)
        list_of_results.append(results)
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
