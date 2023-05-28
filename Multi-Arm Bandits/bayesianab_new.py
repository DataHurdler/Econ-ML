import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

# set the number of bandits
N_bandits = 5
# set the number of trials
# only in demonstration
N = 100000


class BayesianAB:
    """
    This class implements various algorithms for solving the Multi-Armed Bandit problem,
    including epsilon-greedy, optimistic initial values, Upper Confidence Bound (UCB1),
    gradient bandit, and Bayesian bandits.

    Args:
        number_of_bandits (int): The number of bandits. Default is 2.
        number_of_trials (int): The number of trials/visitors. Default is 100,000.
        p_max (float): The maximum win probability for the last bandit. Default is 0.75.
        p_diff (float): The difference between the win probabilities of the bandits. Default is 0.05.
        p_min (float): The minimum win probability for the first (number_of_bandits - 1) bandits. Default is 0.1.

    Raises:
        ValueError: If p_min > p_max - p_diff, indicating an invalid configuration.

    Attributes:
        prob_true (list): True win probabilities of the bandits (only used for demonstration).
        prob_win (list): Estimated win probabilities of the bandits.
        history (list): History of win probability estimates over time.
        history_bandit (list): History of selected bandits over time (for Monte Carlo).
        count (list): Number of times each bandit was selected (only used for demonstration).
        pref (list): Preference values for the bandits (only used for gradient_bandit).
        pi (list): Action probabilities for the bandits (only used for gradient_bandit).
        alpha (list): Alpha parameters of the Beta distribution for Bayesian bandits.
        beta (list): Beta parameters of the Beta distribution for Bayesian bandits.
        N (int): Number of trials/visitors.

    """

    def __init__(
            self,
            number_of_bandits: int = 2,
            number_of_trials: int = 100000,
            p_max: float = 0.75,
            p_diff: float = 0.05,
            p_min: float = 0.1
    ):
        if p_min > p_max - p_diff:
            raise ValueError("Condition p_min < p_max - p_diff not satisfied. Exit...")

        self.prob_true = [0] * number_of_bandits  # True win probabilities of the bandits (only used for demonstration)
        self.prob_win = [0] * number_of_bandits  # Estimated win probabilities of the bandits
        self.history = []  # History of win probability estimates over time
        self.history_bandit = []  # History of selected bandits over time (for Monte Carlo)
        self.count = [0] * number_of_bandits  # Number of times each bandit was selected (only used for demonstration)
        self.pref = [0] * number_of_bandits  # Preference values for the bandits (only used for gradient_bandit)
        self.pi = [1 / number_of_bandits] * number_of_bandits  # Action probabilities for the bandits (only used for gradient_bandit)
        self.alpha = [1] * number_of_bandits  # Alpha parameters of the Beta distribution for Bayesian bandits
        self.beta = [1] * number_of_bandits  # Beta parameters of the Beta distribution for Bayesian bandits
        self.N = number_of_trials  # Number of trials/visitors

        # Set the win rates for the bandits
        self.prob_true[-1] = p_max
        for i in range(0, number_of_bandits - 1):
            self.prob_true[i] = round(p_max - random.uniform(p_diff, p_max - p_min), 2)

    def pull(
            self,
            i,
    ) -> bool:
        """
        Simulates pulling the arm of a bandit.

        Args:
            i (int): The index of the bandit.

        Returns:
            bool: True if the arm yields a win, False otherwise.
        """
        return random.random() < self.prob_true[i]

    def update(
            self,
            i,
            k,
    ) -> None:
        """
        Updates the win probability estimate for a bandit.

        Args:
            i (int): The index of the bandit.
            k (int): The trial/visitor number.

        Returns:
            None
        """
        outcome = self.pull(i)
        self.prob_win[i] = (self.prob_win[i] * k + outcome) / (k + 1)
        self.history.append(self.prob_win.copy())
        self.history_bandit.append(i)  # For Monte Carlo
        self.count[i] += 1

    def epsilon_greedy(
            self,
            epsilon: float = 0.5,
    ):
        """
        Implements the epsilon-greedy algorithm.

        Args:
            epsilon (float): The exploration rate. Default is 0.5.

        Returns:
            list: The history of win probability estimates over time.
        """
        self.history.append(self.prob_win.copy())

        for k in range(1, self.N):
            if random.random() < epsilon:
                i = random.randrange(0, len(self.prob_win))
            else:
                i = np.argmax(self.prob_win)

            self.update(i, k)

        return self.history

    def optim_init_val(
            self,
            init_val: float = 0.99,
    ):
        """
        Implements the optimistic initial values algorithm.

        Args:
            init_val (float): The initial win probability estimate for all bandits. Default is 0.99.

        Returns:
            list: The history of win probability estimates over time.
        """
        self.prob_win = [init_val] * len(self.prob_win)
        self.history.append(self.prob_win.copy())

        for k in range(1, self.N):
            i = np.argmax(self.prob_win)

            self.update(i, k)

        return self.history

    def ucb1(
            self,
            c=1,
    ):
        """
        Implements the Upper Confidence Bound (UCB1) algorithm.

        Args:
            c (float): The exploration parameter. Default is 1.

        Returns:
            list: The history of win probability estimates over time.
        """
        self.history.append(self.prob_win.copy())
        bandit_count = [0.0001] * len(self.prob_win)

        for k in range(1, self.N):
            bound = self.prob_win + c * np.sqrt(np.divide(2 * np.log(k), bandit_count))
            i = np.argmax(bound)

            self.update(i, k)

            if bandit_count[i] < 1:
                bandit_count[i] = 0
            bandit_count[i] += 1

        return self.history

    def gb_update(
            self,
            i,
            k,
            a,
    ):
        """
        Updates the preference values and win probability estimate for the gradient bandit algorithm.

        Args:
            i (int): The index of the bandit.
            k (int): The trial/visitor number.
            a (float): The step size.

        Returns:
            list: The updated preference values.
        """
        outcome = self.pull(i)
        for z in range(len(self.pref)):
            if z == i:
                self.pref[z] = self.pref[z] + a * (outcome - self.prob_win[z]) * (1 - self.pi[z])
            else:
                self.pref[z] = self.pref[z] - a * (outcome - self.prob_win[z]) * self.pi[z]

        self.prob_win[i] = (self.prob_win[i] * k + outcome) / (k + 1)

        return self.pref

    def gradient_bandit(
            self,
            a=0.2,
    ):
        """
        Implements the gradient bandit algorithm.

        Args:
            a (float): The step size. Default is 0.2.

        Returns:
            list: The history of [pi, pref, prob_win] over time.
        """
        self.history.append([self.pi.copy(), self.pref.copy(), self.prob_win.copy()])

        for k in range(1, self.N):
            self.pi = np.exp(self.pref) / sum(np.exp(self.pref))
            pick = random.choices(np.arange(len(self.pref)), weights=self.pi)
            i = pick[0]
            self.pref = self.gb_update(i, k, a)

            self.count[i] += 1
            self.history.append([self.pi.copy(), self.pref.copy(), self.prob_win.copy()])
            self.history_bandit.append(i)  # For Monte Carlo

        return self.history

    def bb_update(
            self,
            a,
            b,
            i,
    ):
        """
        Updates the alpha and beta parameters for the Bayesian bandits.

        Args:
            a (list): The alpha parameters.
            b (list): The beta parameters.
            i (int): The index of the bandit.

        Returns:
            tuple: The updated alpha and beta parameters.
        """
        outcome = self.pull(i)
        a[i] += outcome
        b[i] += 1 - outcome
        self.count[i] += 1

        return a, b

    def bayesian_bandits(
            self,
            sample_size: int = 10,
    ):
        """
        Implements the Bayesian bandits algorithm.

        Args:
            sample_size (int): The number of samples used for estimating the maximum win probability.
                Default is 10.

        Returns:
            list: The history of [a_hist, b_hist] over time.
        """
        a_hist, b_hist = [], []
        a_hist.append(self.alpha.copy())
        b_hist.append(self.beta.copy())

        for k in range(1, self.N):
            max_samples = []
            for i in range(len(self.alpha)):
                samples = np.random.beta(self.alpha[i], self.beta[i], sample_size)
                max_samples.append(max(samples))

            i = np.argmax(max_samples)
            self.alpha, self.beta = self.bb_update(self.alpha, self.beta, i)
            a_hist.append(self.alpha.copy())
            b_hist.append(self.beta.copy())

        self.history = [a_hist, b_hist]
        return self.history


# The following functions are used to plot history
def plot_history(
        history: list,
        prob_true: list,
        col=2,
        k=N,
):
    if type(history[0][0]) == list:  # To accommodate gradient bandit
        df_history = pd.DataFrame([arr[col] for arr in history][:k])
    else:
        df_history = pd.DataFrame(history[:k])

    plt.figure(figsize=(20, 5))

    # Define the color palette
    colors = sns.color_palette("Set2", len(prob_true))

    for i in range(len(prob_true)):
        sns.lineplot(x=df_history.index, y=df_history[i], color=colors[i])

    # Create custom legend using prob_true and colors
    custom_legend = [plt.Line2D([], [], color=colors[i], label=prob_true[i]) for i in range(len(prob_true))]
    plt.legend(handles=custom_legend)

    plt.tight_layout()


def bb_plot_history(
        history: list,
        prob_true: list,
        k=-1,
):
    x = np.linspace(0, 1, 100)
    legend_str = [[]] * len(prob_true)
    plt.figure(figsize=(20, 5))

    for i in range(len(prob_true)):
        a = history[0][k][i]
        b = history[1][k][i]
        y = beta.pdf(x, a, b)
        legend_str[i] = f'{prob_true[i]}, alpha: {a}, beta: {b}'
        plt.plot(x, y)

    plt.legend(legend_str)
    plt.tight_layout()


if __name__ == "__main__":
    random.seed(42)

    # epsilon greedy
    eg = BayesianAB(N_bandits)
    print(f'The true win rates: {eg.prob_true}')
    eg_history = eg.epsilon_greedy(epsilon=0.5)
    print(f'The observed win rates: {[float("{:.4f}".format(x)) for x in eg.prob_win]}')
    print(f'Number of times each bandit was played: {eg.count}')

    # plot the entire experiment history
    plot_history(history=eg.history, prob_true=eg.prob_true)
    # plt.savefig("eg.png")

    # plot history of epsilon greedy after 100 pulls
    plot_history(history=eg.history, prob_true=eg.prob_true, k=100)
    # plt.savefig("eg_100.png")

    # optimistic initial values
    oiv = BayesianAB(N_bandits)
    print(f'The true win rates: {oiv.prob_true}')
    oiv_history = oiv.optim_init_val(init_val=0.99)
    print(f'The observed win rates: {[float("{:.4f}".format(x)) for x in oiv.prob_win]}')
    print(f'Number of times each bandit was played: {oiv.count}')

    # plot the entire experiment history
    plot_history(history=oiv.history, prob_true=oiv.prob_true)

    # plot history of optimistic initial values after 100 pulls
    plot_history(history=oiv.history, prob_true=oiv.prob_true, k=100)

    # Upper Confidence Bound (UCB1)
    ucb = BayesianAB(N_bandits)
    print(f'The true win rates: {ucb.prob_true}')
    ucb_history = ucb.ucb1()
    print(f'The observed win rates: {[float("{:.4f}".format(x)) for x in ucb.prob_win]}')
    print(f'Number of times each bandit was played: {ucb.count}')

    # plot the entire experiment history
    plot_history(history=ucb.history, prob_true=ucb.prob_true)

    # plot history of UCB1 after 100 pulls
    plot_history(history=ucb.history, prob_true=ucb.prob_true, k=100)

    # Gradient bandit
    gb = BayesianAB(N_bandits)
    print(f'The true win rates: {gb.prob_true}')
    gb_history = gb.gradient_bandit()
    print(f'The observed win rates: {[float("{:.4f}".format(x)) for x in gb.prob_win]}')
    print(f'Number of times each bandit was played: {gb.count}')

    # plot the entire experiment history
    plot_history(history=gb.history, prob_true=gb.prob_true)

    # plot history of gradient bandit after 100 pulls
    plot_history(history=gb.history, prob_true=gb.prob_true, k=100)

    # plot preference
    plot_history(history=gb.history, prob_true=gb.prob_true, col=1)

    # plot pi
    plot_history(history=gb.history, prob_true=gb.prob_true, col=0)

    # Bayesian bandits
    bb = BayesianAB(N_bandits)
    print(f'The true win rates: {bb.prob_true}')
    bb_history = bb.bayesian_bandits(sample_size=10)
    print(f'The observed win rates: {np.divide(bb.history[0][-1], bb.count)}')
    print(f'Number of times each bandit was played: {bb.count}')

    # plot the entire experiment history
    bb_plot_history(history=bb.history, prob_true=bb.prob_true)

    # plot history of Bayesian bandits after 100 pulls
    bb_plot_history(history=bb.history, prob_true=bb.prob_true, k=100)

    # plot history of Bayesian bandits after 800 pulls
    bb_plot_history(history=bb.history, prob_true=bb.prob_true, k=800)
