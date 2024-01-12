import os
import random
import pandas as pd
from time import perf_counter

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
    "learning_rate" : (0.000_000_1, 0.000_01),
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
        print(optimizer.space, '\n', optimizer.res, '\n', optimizer.max)

    logger = JSONLogger(path=f"Results\\Hyperparameter_testing\\{number_epochs}_epochs\\BO_logs.log")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=60
    )

    print(optimizer.max)
    results = pd.json_normalize(optimizer.res)
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


if __name__ == "__main__":
    # Set # epochs from run function
    no_epochs = 128

    if not os.path.isdir("Results\\Hyperparameter_testing"):
        os.makedirs("Results\\Hyperparameter_testing")
    if not os.path.isdir(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs"):
        os.makedirs(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs")

    start = perf_counter()
    results = run_BO(BO_HParameters=pspace_BO, number_epochs=128, load_results=True)  # Make sure to change load
    results.to_csv(f"Results\\Hyperparameter_testing\\{no_epochs}_epochs\\Bayesian_results")
    BO_time = perf_counter()

    # TODO - check if I can use the knwoledge from the ranomd tests as a starting pad
    print(f"BO took {round((BO_time - start)/60, 2)} mins to do method 10 times")

    # run_GPyOPT(GPy_HParameters=pspace_GPy)  # Took ~ 60% longer and 3.11 + multiprocessing failed
    # print(f"GPy took {round((perf_counter() - BO_time)/60, 2)} mins to do method 5 times")