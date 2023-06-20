import random
import matplotlib.pyplot as plt


class PrisonersDilemma:
    """Class representing the Prisoner's Dilemma game."""

    def __init__(
            self,
            payoffs,
            num_agents: int = 100,
            num_episodes: int = 10000,
            exploration_rate: float = 0.05,
            trial_ratio: float = 0.01,
    ):
        """
        Initialize the PrisonersDilemma instance.

        Args:
            payoffs (list): Payoff values for different choices in the game.
            num_agents (int): Number of agents in the game.
            num_episodes (int): Number of episodes to run the game.
            exploration_rate (float): Exploration rate for agents' action selection.
        """
        assert num_agents % 2 == 0 # check num_agents is an even number
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.exploration_rate = exploration_rate
        self.trial_ratio = trial_ratio

        # Define the rewards for different choices
        s, p, r, t = payoffs
        self.rewards = {
            ('LEFT', 'LEFT'): (p, p),
            ('RIGHT', 'RIGHT'): (r, r),
            ('LEFT', 'RIGHT'): (t, s),
            ('RIGHT', 'LEFT'): (s, t)
        }

        # Define the initial accumulated rewards for each agent's strategy
        self.rewards_accumulated = dict()
        self.rewards_history = dict()
        self.strategies_history = dict()
        for agent_id in range(num_agents):
            self.rewards_accumulated[agent_id] = {
                'LEFT': 0,
                'RIGHT': 0
            }
            self.rewards_history[agent_id] = list()
            self.strategies_history[agent_id] = list()

        # Define a list to track the percentage of LEFT played in each round
        self.left_percentage_rounds = list()

        # Define a list to store the top 5 agents' IDs
        self.top_agents = list()

    def q_learning(
            self,
            agent_rewards,
            opponent_rewards,
    ) -> tuple[str, str]:
        """
        Perform Q-learning to determine the actions of an agent and its opponent.

        Args:
            agent_rewards (dict): Accumulated rewards for the agent's strategy.
            opponent_rewards (dict): Accumulated rewards for the opponent's strategy.

        Returns:
            tuple[str, str]: Chosen actions for the agent and opponent.
        """
        agent_action = max(agent_rewards,
                           key=agent_rewards.get) if random.random() > self.exploration_rate else random.choice(
            ['LEFT', 'RIGHT'])
        opponent_action = max(opponent_rewards,
                              key=opponent_rewards.get) if random.random() > self.exploration_rate else random.choice(
            ['LEFT', 'RIGHT'])

        return agent_action, opponent_action

    def run_single_episode(
            self,
            algo: str,
            episode: int,
    ) -> None:
        """
        Run a single episode of the game.

        Args:
            algo (str): The algorithm used to pick action
            episode (int): Episode number.
        """
        # Track the number of agents choosing LEFT in the current round
        left_count = 0

        # Define a list to store the agent_id already played
        already_played = []

        # Iterate over all agents
        for agent_id in range(self.num_agents):
            if agent_id in already_played:
                # agent has already played in this episode
                continue

            # Select opponent agent randomly
            opponent_id = random.choice(list(range(agent_id + 1, self.num_agents)))

            while (opponent_id == agent_id) or (opponent_id in already_played):
                opponent_id = random.choice(list(range(agent_id + 1, self.num_agents)))

            already_played.extend([agent_id, opponent_id])

            # Retrieve the accumulated rewards for the agent and opponent
            agent_rewards, opponent_rewards = self.rewards_accumulated[agent_id], self.rewards_accumulated[opponent_id]

            # Select actions for the agent and opponent based on accumulated rewards and exploration rate
            if episode < round(self.trial_ratio * self.num_episodes):
                if episode == 0:
                    agent_action, opponent_action = ['RIGHT', 'RIGHT']
                else:
                    agent_action = random.choice(['LEFT', 'RIGHT'])
                    opponent_action = random.choice(['LEFT', 'RIGHT'])
            else:
                if hasattr(self, algo):
                    algorithm = getattr(self, algo)
                    agent_action, opponent_action = algorithm(agent_rewards, opponent_rewards)
                else:
                    raise ValueError("Invalid method.")
            # Get the rewards for the chosen actions
            reward_agent, reward_opponent = self.rewards[(agent_action, opponent_action)]

            # Store rewards and strategies of the two agents that played
            self.rewards_history[agent_id].append(reward_agent)
            self.rewards_history[opponent_id].append(reward_opponent)
            self.strategies_history[agent_id].append(agent_action)
            self.strategies_history[opponent_id].append(opponent_action)

            # update average rewards from each action/strategy
            self.rewards_accumulated[agent_id][agent_action] = update_mean(
                self.rewards_accumulated[agent_id][agent_action],
                reward_agent,
                self.strategies_history[agent_id].count(agent_action)
            )

            self.rewards_accumulated[opponent_id][opponent_action] = update_mean(
                self.rewards_accumulated[opponent_id][opponent_action],
                reward_opponent,
                self.strategies_history[opponent_id].count(opponent_action)
            )

            # Increment the left_count if agent chooses LEFT
            if agent_action == 'LEFT':
                left_count += 1
            if opponent_action == 'LEFT':
                left_count += 1

        # Calculate the percentage of LEFT being played in the current round
        left_percentage = (left_count / self.num_agents) * 100
        self.left_percentage_rounds.append(left_percentage)

        assert len(already_played) == self.num_agents

    def run_multi_episodes(self, algo: str) -> None:
        """Run multiple episodes of the game."""
        for episode in range(self.num_episodes):
            self.run_single_episode(algo, episode)

    def plot_all(self) -> None:
        """Plot the percentage of LEFT played in each round and save the plot as 'all.png'."""
        plt.plot(range(1, self.num_episodes + 1), self.left_percentage_rounds, linestyle="", marker="o", markersize=2)
        plt.xlabel('Round')
        plt.ylabel('Percentage of LEFT')
        plt.title('Percentage of LEFT Played in Each Round')
        plt.savefig('all.png', dpi=150)
        plt.show()

        agent_totals = [sum(self.rewards_history[agent_id]) for agent_id in range(self.num_agents)]
        plt.plot(range(self.num_agents), agent_totals)
        plt.savefig('payoffs.png', dpi=150)
        plt.show()

    def top_bottom_agents(self):
        """Print information about the top and bottom 5 agents based on accumulated rewards."""
        top_agents = sorted(range(self.num_agents),
                            key=lambda agent_id: sum(self.rewards_history[agent_id]), reverse=True)[:5]
        bottom_agents = sorted(range(self.num_agents),
                               key=lambda agent_id: sum(self.rewards_history[agent_id]), reverse=False)[:5]

        for i, agent_id in enumerate(top_agents + bottom_agents):
            is_top_agent = agent_id in top_agents
            agents_info(self.strategies_history, self.rewards_history, agent_id, is_top_agent)


def update_mean(previous_mean, new_value, n_episodes) -> float:
    """
    Update the mean value based on a new value and the number of episodes.

    Args:
        previous_mean (float): Previous mean value.
        new_value (float): New value to be incorporated into the mean.
        n_episodes (int): Number of episodes.

    Returns:
        float: Updated mean value.
    """
    return previous_mean + (new_value - previous_mean) / (n_episodes + 1)


def agents_info(
        strategies_history,
        rewards_history,
        agent_id,
        is_top_agent: bool,
) -> None:
    """
    Print information about an agent's strategies and rewards.

    Args:
        strategies_history (dict): Dictionary containing agents' strategy histories.
        rewards_history (dict): Dictionary containing agents' reward histories.
        agent_id (int): ID of the agent.
        is_top_agent (bool): Flag indicating whether the agent is a top agent or bottom agent.
    """
    left_count = strategies_history[agent_id].count('LEFT')
    right_count = strategies_history[agent_id].count('RIGHT')
    left_total = 0
    for reward, strategy in zip(rewards_history[agent_id], strategies_history[agent_id]):
        if strategy == 'LEFT':
            left_total += reward
    right_total = 0
    for reward, strategy in zip(rewards_history[agent_id], strategies_history[agent_id]):
        if strategy == 'RIGHT':
            right_total += reward
    total_score = left_total + right_total

    agent_type = "Top" if is_top_agent else "Bottom"
    print(f"{agent_type} Agent (ID: {agent_id}): Total Score = {total_score}, \n"
          f"LEFT Count = {left_count}, RIGHT Count = {right_count}, \n"
          f"LEFT Rewards = {left_total}, RIGHT Rewards = {right_total}, \n"
          f"LEFT Average = {left_total / left_count:.4f}, RIGHT Average = {right_total / right_count:.4f}")

    print(strategies_history[agent_id])


if __name__ == "__main__":
    PAYOFFS = [0, 1, 3, 5]
    # Define the number of agents
    N_AGENTS = 100
    # Number of episodes
    N_EPISODES = 10000
    # Define the exploration rate
    EXP_RATE = 0.02
    # Define length of trial period in the beginning
    TRIAL_RATIO = 0.05

    random.seed(42)

    PD = PrisonersDilemma(PAYOFFS, N_AGENTS, N_EPISODES, EXP_RATE, TRIAL_RATIO)
    PD.run_multi_episodes('q_learning')
    PD.plot_all()
    PD.top_bottom_agents()
