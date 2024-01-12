import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Bayesian Packages to use - HyperOpt, GPyOpt, PyMC, BayesianOptimization, Ax (Facebook's tool)
from GPyOpt.methods import BayesianOptimization  # Had Installation issues - built from setup.py + upgrade cython
# from hyperopt import hp, fmin, tpe, space_eval  # HyperOpt does not use gaussian process, but Tree of Parzen Estimators (TPE method)
from bayes_opt import BayesianOptimization as BO  # Import as to clear namespace - tested both packages
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# Import black box functions
from Hparameter_tuning_random_approach import test_model
from NNQmodel import NNQPlayer


# Define paremter space
pspace_BO = {
    "learning_rate": (0.000_000_1, 0.000_01),
    "gamma": (0.90, 0.99),
    "reward_for_all_dice": (0, 10),
    "reward_factor_for_initial_dice_picked": (0, 1),
    "reward_factor_for_picking_choice_correctly": (0, 10),
    "batch_size": (32, 128),
    "buffer_size": (32, 128),
    "memory": (2_000, 10_000),
}


def run_BO(BO_HParameters, number_epochs, load_results=False):
    optimizer = BO(  # TODO test the bounds minimiser object!
        f=test_model,
        pbounds=BO_HParameters,
        verbose=1,
        random_state=random.randint(0, 20),
        allow_duplicate_points=True  # Just easier to mark thi true as I'm piece-meal mapping the space
    )

    if load_results and os.path.isfile(f"Results\\Hyperparameter_testing\\{number_epochs}_epochs\\BO_logs.log.json"):
        # Create a backup with the appended file so as to not lose points when resetting
        # This is necessary because the JSONlogger object only logs seen points and will overwrite the object when 
        # initialised below
        with open(f"Results\\Hyperparameter_testing\\{number_epochs}_epochs\\BO_logs_backup.log.json", 'a') as backup:
            with open(f"Results\\Hyperparameter_testing\\{number_epochs}_epochs\\BO_logs.log.json", 'r') as original:
                print(original.__dict__)
                print(original.readlines)
                backup.writelines(original.readlines())

        load_logs(optimizer, logs=[f"Results\\Hyperparameter_testing\\{number_epochs}_epochs\\BO_logs_backup.log.json"])
        # See this - https://www.vidensanalytics.com/nouveau-blog/bayesian-optimization-to-the-rescue can load all logs 
        # in a list!
        print("Loaded previous points:\n")
        print("Space: \n", optimizer.space, '\nres: \n', optimizer.res, '\nmax:\n', optimizer.max)

    logger = JSONLogger(path=f"Results\\Hyperparameter_testing\\{number_epochs}_epochs\\BO_logs.log")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=1,
        n_iter=10
    )

    print(optimizer.max)
    results = pd.json_normalize(optimizer.res)
    results.columns = results.columns.str.removeprefix("params.")
    return results


pspace_GPy = [
    {"name": "learning_rate", "type": "continuous", "domain": (0.000_000_1, 0.000_01)},
    {"name": "gamma", "type": "continuous", "domain": (0.90, 0.99)},
    {"name": "reward_for_all_dice", "type": "continuous", "domain": (0, 10)},
    {"name": "reward_factor_for_initial_dice_picked", "type": "continuous", "domain": (0, 1)},
    {"name": "reward_factor_for_picking_choice_correctly", "type": "continuous", "domain": (0, 10)},
    {"name": "batch_size", "type": "continuous", "domain": (32, 128)},
    {"name": "buffer_size", "type": "continuous", "domain": (32, 128)},
    {"name": "memory", "type": "continuous", "domain": (2_000, 10_000)}  
]


def test_model_GPy_API(X):
    learning_rate = X[:, 0]
    gamma = X[:, 1]
    reward_for_all_dice = X[:, 2]
    reward_factor_for_initial_dice_picked = X[:, 3]
    reward_factor_for_picking_choice_correctly = X[:, 4]
    batch_size = X[:, 5]
    buffer_size = X[:, 6]
    memory = X[:, 7]

    # TODO add in architecture
    model = NNQPlayer(
        show_figures=False,
        learning_rate=learning_rate,
        gamma=gamma,
        reward_for_all_dice=reward_for_all_dice,
        reward_factor_for_initial_dice_picked=reward_factor_for_initial_dice_picked,
        reward_factor_for_picking_choice_correctly=reward_factor_for_picking_choice_correctly,
        batch_size=int(batch_size),  # Note these need to be integers - coerce them to integers here for Bayesian Opt
        length_of_memory=int(memory),
        buffer_size=int(buffer_size),
        name="HP_testing_GPyOpt"
    )

    model.run(4, 64, save_results=False, save_model=False, verbose=False)

    return model.average_score


def run_GPyOPT(GPy_HParameters):
    # batch_size = 4
    # num_cores = 4

    myBopt = BayesianOptimization(
        f=test_model_GPy_API,
        domain=GPy_HParameters,
        # num_cores=num_cores,
        # batch_size=batch_size
        )
    myBopt.run_optimization(max_iter=10)
    myBopt.plot_acquisition()


# Plotting and visualisation functions defined below 
def identify_top_correlates(resluts: pd.DataFrame):
    """Returns the most important, second most important, third most important set of 3 parameters. 
    REturns a list of pd.index"""
    correlates_with_target = results.corr()["target"]
    correlates_with_target.drop(index="target", inplace=True)
    correlates_with_target = correlates_with_target.apply(lambda x: abs(x)).sort_values(ascending=False)
    return [correlates_with_target.index[0:2], correlates_with_target.index[1:3],  correlates_with_target.index[2:4]]


def three_d_map_of_specified_parameters(results: pd.DataFrame(), top_parameters: pd.Index, no_epochs: int):
    x = results[top_parameters[0]].to_numpy()
    y = results[top_parameters[1]].to_numpy()

    Z = results["target"].to_numpy()

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    name = top_parameters[0] + "_and_" + top_parameters[1] + "_plotted_against_target"
    plt.savefig(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs\\{name}.png")
    plt.close()

    return 


def plot_correlation_heatmap(results: pd.DataFrame(), no_epochs: int):
    heatmap = sns.heatmap(results.corr(), annot=True)
    # heatmap.fig.suptitle('HeatMAp of HyperParameters correlation with Avg_Score', y=1.02, fontsize=16)
    plt.show()
    plt.savefig(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs\\Hyperparameter_heatmap.png")
    plt.close()
    return heatmap


def plot_hyperparameter_space(results: pd.DataFrame, no_epochs: int):
    """Plot the results from hyperparameter testing to identify possible configurations to use

    :param results: the dataframe with cols of HyperParameters and the average score of the trained model
    :type results: pd.DataFrame
    """
    vars = results.drop(columns="target").columns.to_list()
    pair_plot = sns.pairplot(results, vars=vars, hue="target")

    pair_plot.fig.suptitle('Pair Plot of HyperParameters with Color by Avg_Score', y=1.02, fontsize=16)
    plt.show()
    plt.savefig(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs\\Hyperparameter_testing_pairplot_results.png") 
    plt.close()
    return


if __name__ == "__main__":
    # Set # epochs from run function
    no_epochs = 128

    if not os.path.isdir("Results\\Hyperparameter_testing"):
        os.makedirs("Results\\Hyperparameter_testing")
    if not os.path.isdir(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs"):
        os.makedirs(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs")

    start = perf_counter()
    results = run_BO(BO_HParameters=pspace_BO, number_epochs=128, load_results=True)  # Make sure to change load
    results.to_csv(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs\\Bayesian_results.csv", index=False)
    BO_time = perf_counter()

    # TODO - check if I can use the knwoledge from the ranomd tests as a starting pad
    print(f"BO took {round((BO_time - start)/60, 2)} mins to do method 10 times")

    # run_GPyOPT(GPy_HParameters=pspace_GPy)  # Took ~ 60% longer and 3.11 + multiprocessing failed
    # print(f"GPy took {round((perf_counter() - BO_time)/60, 2)} mins to do method 5 times")

    # Plotting
    top_correlates = identify_top_correlates(results)
    for correlates in top_correlates:
        three_d_map_of_specified_parameters(results=results, top_parameters=correlates, no_epochs=no_epochs)

    plot_correlation_heatmap(results=results, no_epochs=no_epochs)
    plot_hyperparameter_space(results=results, no_epochs=no_epochs)