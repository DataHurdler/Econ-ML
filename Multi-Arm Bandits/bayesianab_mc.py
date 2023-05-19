import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

# set the number of bandits
N_bandits = 5
# set the number of visitors
N = 10001
# set the number of trials to try all bandits
N_start = 50
# set the number of trials
M = 1000


class BayesianAB:
    def __init__(
            self,
            number_of_bandits: int = 2,
            p_max: float = .75,
            p_diff: float = .05,
            p_min: float = .1
    ):
        if p_min > p_max - p_diff:
            warnings.warn("Condition p_min < p_max - p_diff not satisfied. Exit...", UserWarning)
            quit()

        self.prob_true = [0] * number_of_bandits  # only in demonstration
        self.prob_win = [0] * number_of_bandits
        self.history = []
        self.history_bandit = []  # for Monte Carlo
        self.count = [0] * number_of_bandits  # only in demonstration
        # a and b are for bayesian_bandits only
        self.alpha = [1] * number_of_bandits
        self.beta = [1] * number_of_bandits
        # preference and pi are for gradient_bandit only
        self.pref = [0] * number_of_bandits
        self.pi = [1 / number_of_bandits] * number_of_bandits

        # set the last bandit to have a win rate of 0.75 and the rest lower
        # only in demonstration
        self.prob_true[-1] = p_max
        for i in range(0, number_of_bandits - 1):
            self.prob_true[i] = round(p_max - random.uniform(p_diff, p_max - p_min), 2)

    # Receives a random value of 0 or 1
    # only in demonstration
    def pull(
            self,
            i,
    ) -> bool:
        return random.random() < self.prob_true[i]

    # Updates the mean
    def update(
            self,
            i,
            k,
    ):
        outcome = self.pull(i)
        # may use a constant discount rate to discount past
        self.prob_win[i] = (self.prob_win[i] * k + outcome) / (k + 1)
        self.history.append(self.prob_win.copy())
        self.history_bandit.append(i)  # for Monte Carlo
        self.count[i] += 1

    ####################
    # epsilon greedy
    def epsilon_greedy(
            self,
            epsilon: float = 0.5,
    ):

        self.history.append(self.prob_win.copy())

        for k in range(1, N):
            if random.random() < epsilon:
                i = random.randrange(0, len(self.prob_win))
            else:
                # find index of the largest value in prob_win
                i = np.argmax(self.prob_win)

            self.update(i, k)

        return self.history_bandit

    ####################
    # optimistic initial values
    def optim_init_val(
            self,
            init_val: float = 0.99,
    ):

        self.prob_win = [init_val] * len(self.prob_win)
        self.history.append(self.prob_win.copy())

        for k in range(1, N):
            # find index of the largest value in prob_win
            i = np.argmax(self.prob_win)

            self.update(i, k)

        return self.history_bandit

    ####################
    # upper confidence bound (UCB1)
    def ucb1(
            self,
            c=1,
    ):

        self.history.append(self.prob_win.copy())
        bandit_count = [0.0001] * len(self.prob_win)
        # bound = [0] * len(self.prob_win)

        for k in range(1, N):
            bound = self.prob_win + c * np.sqrt(2 * np.log(k) / bandit_count)
            # find index of the largest value in bound
            i = np.argmax(bound)

            self.update(i, k)

            if bandit_count[i] < 1:
                bandit_count[i] = 0
            bandit_count[i] += 1

        return self.history_bandit

    ####################
    # gradient_bandit update
    def gb_update(
            self,
            i,
            k,
            a,
    ):

        outcome = self.pull(i)
        for z in range(len(self.pref)):
            if z == i:
                self.pref[z] = self.pref[z] + a * (outcome - self.prob_win[z]) * (1 - self.pi[z])
            else:
                self.pref[z] = self.pref[z] - a * (outcome - self.prob_win[z]) * self.pi[z]

        self.prob_win[i] = (self.prob_win[i] * k + outcome) / (k + 1)

        return self.pref

    # gradient bandit algorithm
    def gradient_bandit(
            self,
            a=0.2,
    ):

        self.history.append([self.pi.copy(),
                             self.pref.copy(),
                             self.prob_win.copy()])

        for k in range(1, N):
            self.pi = np.exp(self.pref) / sum(np.exp(self.pref))
            pick = random.choices(np.arange(len(self.pref)), weights=self.pi)
            i = pick[0]
            self.pref = self.gb_update(i, k, a)

            self.count[i] += 1
            self.history.append([self.pi.copy(),
                                 self.pref.copy(),
                                 self.prob_win.copy()])
            self.history_bandit.append(i)  # for Monte Carlo

        return self.history_bandit

    ####################
    # bayesian_bandits sample
    def bb_sample(
            self,
            a: int,  # alpha
            b: int,  # beta
            sample_size: int = 10,
    ):

        return np.random.beta(a, b, sample_size)

    # bayesian_bandits update
    def bb_update(
            self,
            a,
            b,
            i,
    ):

        outcome = self.pull(i)
        # may use a constant discount rate to discount past
        a[i] += outcome
        b[i] += 1 - outcome
        self.count[i] += 1

        return a, b

    # Bayesian bandits
    # For Bernoulli distribution, the conjugate prior is Beta distribution
    def bayesian_bandits(
            self,
            sample_size: int = 10,
    ):

        a_hist, b_hist = [], []
        a_hist.append(self.alpha.copy())
        b_hist.append(self.beta.copy())

        for k in range(1, N):
            sample_max = []

            for m in range(len(self.prob_true)):
                m_max = np.max(self.bb_sample(self.alpha[m], self.beta[m], sample_size))
                sample_max.append(m_max.copy())

            i = np.argmax(sample_max)

            self.alpha, self.beta = self.bb_update(self.alpha, self.beta, i)
            a_hist.append(self.alpha.copy())
            b_hist.append(self.beta.copy())
            self.history_bandit.append(i)  # for Monte Carlo

        self.history = [a_hist, b_hist]
        return self.history_bandit


def worker(algo, N_bandits, p_max, p_diff, p_min, n):
    BayesianAB_instance = BayesianAB(N_bandits, p_max, p_diff, p_min)
    return getattr(BayesianAB_instance, algo)()


def monte_carlo(
        algos,
        n=500,
        p_max: float = .75,
        p_diff: float = .05,
        p_min: float = .1
):
    algos_hist = {algo: [] for algo in algos}

    for algo in algos:
        print(f'Running {algo}...')
        with Pool(cpu_count()) as pool:
            func = partial(worker, algo, N_bandits, p_max, p_diff, p_min)
            results = list(pool.imap(func, range(n)))

        algos_hist[algo] = results

    return algos_hist


def run_monte_carlo(
        algos,
        M,
        p_values,
):
    trials = {}
    df_all = {}

    for i in range(len(p_values)):
        print(f'The p_values are {p_values[i]}')
        trials[f'p{i}'] = monte_carlo(algos,
                                      M,
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

    df_all = run_monte_carlo(algos, M, p_values)

    plot_monte_carlo(df_all, algos, 3, 4)
