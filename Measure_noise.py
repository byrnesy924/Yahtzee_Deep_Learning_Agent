import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import perf_counter
from pathlib import Path
from multiprocessing import pool

from Hparameter_tuning_random_approach import test_model
# from NNQmodel import NNQPlayer


model_hyperparameters = {
    "epochs": 16,
    "learning_rate": 0.000_2,
    "gamma": 0.92,
    "reward_for_all_dice": 6.5,
    "reward_factor_for_initial_dice_picked": 0.22,
    "reward_factor_for_picking_choice_correctly": 6.26,
    "reward_factor_total_score": 2,
    "reward_factor_chosen_score": 3.8,
    "punish_amount_for_incorrect_score_choice": -4,
    "punish_for_not_picking_dice": -0.4,
    "batch_size": 101,
    "buffer_size": 80,
    "memory": 6200,
    "name": "Measure_Noise_of_Model_runs"
}


def wrapper_function(runs: int, kwargs: dict):
    results = [test_model(**kwargs) for i in range(runs)]
    return results


if __name__ == "__main__":
    noise_measurement_path = Path("Results/Noise_measurements")
    epochs_path = Path(f"Results/Noise_measurements/{model_hyperparameters['epochs']}_epochs")
    results_path = Path(f"Results/Noise_measurements/{model_hyperparameters['epochs']}_epochs/Noise_Measurements.csv")

    load_results = True
    if not os.path.isdir(noise_measurement_path):
        os.makedirs(noise_measurement_path)
    if not os.path.isdir(epochs_path):
        os.makedirs(epochs_path)

    start = perf_counter()

    no_processes = os.cpu_count() - 10
    args = [(100, model_hyperparameters) for i in range(no_processes)]

    with pool.Pool(no_processes) as pool:
        results_list = [res for res in pool.starmap(wrapper_function, args)]

    results = []
    for item in results_list:
        results.extend(item)
    results_df = pd.DataFrame(results, columns=["Target"])

    print(f"Took {perf_counter() - start}s to run and do {model_hyperparameters['epochs']} epochs with ",
          f"{no_processes} processes")

    fig, ax = plt.subplots(2, 2, figsize=(30, 30))

    sns.histplot(data=results_df["Target"], kde=True, ax=ax[0, 0])
    sns.swarmplot(data=results_df["Target"], size=5, ax=ax[0, 1])
    sns.boxplot(data=results_df["Target"], ax=ax[1, 0])
    sns.violinplot(data=results_df["Target"], ax=ax[1, 1])
    fig.savefig(epochs_path / "Visualisation.png")

    plt.show()
    plt.close()

    results.to_csv(results_path)
