import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from functools import partial

from bayesianab import BayesianAB

# set the number of bandits
N_bandits = 5
# set the number of visitors
N = 10001
# set the number of trials
M = 1000


def worker(algo, N_bandits, N, p_max, p_diff, p_min, n):
    bayesianab_instance = BayesianAB(N_bandits, N, p_max, p_diff, p_min)
    getattr(bayesianab_instance, algo)()
    return bayesianab_instance.history_bandit


def monte_carlo(
        algos,
        n=500,
        N=10001,
        p_max: float = .75,
        p_diff: float = .05,
        p_min: float = .1
):
    algos_hist = {algo: [] for algo in algos}

    for algo in algos:
        print(f'Running {algo}...')
        with Pool(cpu_count()) as pool:
            func = partial(worker, algo, N_bandits, N, p_max, p_diff, p_min)
            results = list(pool.imap(func, range(n)))

        algos_hist[algo] = results

    return algos_hist


def run_monte_carlo(
        algos,
        M,
        N,
        p_values,
):
    trials = {}
    df_all = {}

    for i in range(len(p_values)):
        print(f'The p_values are {p_values[i]}')
        trials[f'p{i}'] = monte_carlo(algos,
                                      M,
                                      N,
                                      p_values[i][0],
                                      p_values[i][1],
                                      p_values[i][2],)

    for i in range(len(p_values)):
        df = pd.DataFrame()
        for j in algos:
            list = [0] * (N - 1)
            for k in range(M):
                list = np.array(list) + np.array([1 if x == 4 else 0 for x in trials[f'p{i}'][j][k]])
            df[j] = (list / M).tolist()

        df_all[f'p{i}'] = df.copy()

    return df_all


def plot_monte_carlo(
        df_all,
        algos,
        col,
        row,
):
    figure, axis = plt.subplots(row, col, figsize=(20, 10))
    colors = sns.color_palette("Set2", len(algos))

    m = 0  # column index
    n = 0  # row index

    for key in df_all:
        ax = axis[n, m]
        for i in range(len(algos)):
            sns.lineplot(x=df_all[key].index, y=df_all[key][algos[i]], linewidth=0.5, color=colors[i], ax=ax)

        ax.set_ylabel('')
        ax.set_title(p_values[n * 3 + m])
        ax.set_xticks([])

        if m == 2:
            # Create custom legend using prob_true and colors
            custom_legend = [plt.Line2D([], [], color=colors[i], label=algos[i]) for i in range(len(algos))]
            ax.legend(handles=custom_legend, loc='upper left', fontsize=9)
            n += 1
            m = 0
        else:
            m += 1

    figure.suptitle('Comparing 5 Algorithms in 12 Different Win Rate Specifications', fontsize=16)

    # Adjust the spacing between subplots
    plt.tight_layout()

    plt.savefig("comparison.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    algos = ['epsilon_greedy', 'optim_init_val', 'ucb1', 'gradient_bandit', 'bayesian_bandits']
    p_values = [
        [.35, .1, .1],
        [.35, .05, .1],
        [.35, .01, .1],
        [.75, .1, .1],
        [.75, .05, .1],
        [.75, .01, .1],
        [.75, .1, .62],
        [.75, .05, .62],
        [.75, .01, .62],
        [.95, .1, .82],
        [.95, .05, .82],
        [.95, .01, .82],
    ]

    df_all = run_monte_carlo(algos, M, N, p_values)

    plot_monte_carlo(df_all, algos, 3, 4)
