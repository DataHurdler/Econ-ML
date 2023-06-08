import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, cpu_count
from tree_ensembles import run_tree_ensembles

plt.ion()


def run_monte_carlo():
    n_individuals_range = range(1000, 100001, 1000)

    all_results = {}

    with Pool(cpu_count()) as pool:
        results = pool.starmap(run_tree_ensembles, [
            (5, n_individuals, 10, False, False, [True]) for n_individuals in n_individuals_range
        ])

    for n_individuals, result in zip(n_individuals_range, results):
        all_results[n_individuals] = result

    return all_results


if __name__ == "__main__":
    mc_output = run_monte_carlo()

    df_results = pd.DataFrame(mc_output)
    df_results.to_csv(mc_output.csv, index=False)
