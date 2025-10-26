from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Bandit(ABC):
    """Abstract base class defining the Bandit interface."""

    @abstractmethod
    def __init__(self, p:list) -> None:
        """Initialize the bandit with true rewards."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the class."""
        pass

    @abstractmethod
    def pull(self, arm:int) -> float:
        """Simulate pulling an arm.

        Args:
          arm(int):

        Returns:

        """
        pass

    @abstractmethod
    def update(self, arm:int, reward:float) -> None:
        """Update internal statistics after pulling an arm.

        Args:
          arm(int):
          reward(float):

        Returns:

        """
        pass

    @abstractmethod
    def experiment(self) -> None:
        """Run the complete experiment for the bandit."""
        pass

    @abstractmethod
    def report(self) -> pd.DataFrame:
        """Report final results and save to CSV."""
        pass


class EpsilonGreedy(Bandit):
    """Implements the Epsilon-Greedy bandit algorithm."""

    def __init__(self, rewards:list, n_trials:int) -> None:
        """
        Initialize Epsilon-Greedy algorithm.

        Args:
            rewards(list): True mean rewards of each arm.
            n_trials(int): Number of trials to run.
        """
        self.true_rewards = np.array(rewards)
        self.n_trials = n_trials
        self.k = len(rewards)
        self.estimates = np.zeros(self.k)
        self.counts = np.zeros(self.k)
        self.data = []

    def __repr__(self) -> str:
        return f"EpsilonGreedy(k={self.k}, n_trials={self.n_trials})"

    def pull(self, arm:int) -> float:
        """Simulate pulling an arm.

        Args:
          arm(int): The index of the arm to pull.

        Returns:
          float: Observed stochastic reward.

        """
        reward = np.random.normal(self.true_rewards[arm], 0.2)
        return reward

    def update(self, arm:int, reward:float) -> None:
        """Update estimated reward for an arm.

        Args:
          arm(int): The index of the arm pulled.
          reward(float): The observed reward.

        Returns:

        """
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

    def experiment(self) -> None:
        """Run Epsilon-Greedy experiment with decaying epsilon."""
        optimal_mean = max(self.true_rewards)
        cumulative_reward = 0
        cumulative_regret = 0

        for t in range(1, self.n_trials + 1):
            epsilon = max(0.05, 1 / t)  # decaying epsilon
            if np.random.rand() < epsilon:
                arm = np.random.randint(0, self.k)
            else:
                arm = np.argmax(self.estimates)

            reward = self.pull(arm)
            self.update(arm, reward)
            cumulative_reward += reward
            regret = optimal_mean - self.true_rewards[arm]
            cumulative_regret += regret

            self.data.append(
                [t, arm, reward, cumulative_reward, cumulative_regret, "EpsilonGreedy"]
            )

    def report(self) -> pd.DataFrame:
        """Save results to CSV and log summary.

        Args:

        Returns:
          pd.DataFrame: Experiment results.

        """
        df = pd.DataFrame(
            self.data,
            columns=[
                "Trial",
                "Bandit",
                "Reward",
                "CumulativeReward",
                "CumulativeRegret",
                "Algorithm",
            ],
        )
        df.to_csv("results_epsilon_greedy.csv", index=False)
        avg_reward = df["Reward"].mean()
        total_regret = df["CumulativeRegret"].iloc[-1]
        logger.info(
            f"[Epsilon-Greedy] average reward: {avg_reward:.3f} | "
            f"Cumulative regret: {total_regret:.3f}"
        )
        return df


class ThompsonSampling(Bandit):
    """Implements the Thompson Sampling bandit algorithm."""

    def __init__(self, rewards:list, n_trials:int, precision:float=1.0) -> None:
        """
        Initialize Thompson Sampling algorithm.

        Args:
            rewards(list): True mean rewards of each arm.
            n_trials(int): Number of trials to run.
            precision(float): Known precision (1/variance) of reward noise.
        """
        self.true_rewards = np.array(rewards)
        self.n_trials = n_trials
        self.k = len(rewards)
        self.precision = precision
        self.means = np.zeros(self.k)
        self.lambdas = np.ones(self.k) * 1e-3  # prior precision
        self.data = []

    def __repr__(self) -> str:
        return f"ThompsonSampling(k={self.k}, n_trials={self.n_trials})"

    def pull(self, arm:int) -> float:
        """Simulate pulling an arm.

        Args:
          arm(int): The index of the arm to pull.

        Returns:
          float: Observed stochastic reward.

        """
        reward = np.random.normal(self.true_rewards[arm], 0.2)
        return reward

    def update(self, arm:int, reward:float) -> None:
        """Bayesian posterior update for Gaussian likelihood with known precision.

        Args:
          arm(int): The index of the arm pulled.
          reward(float): The observed reward.

        Returns:

        """
        self.lambdas[arm] += self.precision
        self.means[arm] = (
            self.means[arm] * (self.lambdas[arm] - self.precision) + self.precision * reward
        ) / self.lambdas[arm]

    def experiment(self) -> None:
        """Run Thompson Sampling experiment."""
        optimal_mean = max(self.true_rewards)
        cumulative_reward = 0
        cumulative_regret = 0

        for t in range(1, self.n_trials + 1):
            samples = np.array([np.random.normal(self.means[i], 1 / np.sqrt(self.lambdas[i])) for i in range(self.k)])
            arm = np.argmax(samples)
            reward = self.pull(arm)
            self.update(arm, reward)
            cumulative_reward += reward
            regret = optimal_mean - self.true_rewards[arm]
            cumulative_regret += regret

            self.data.append(
                [t, arm, reward, cumulative_reward, cumulative_regret, "ThompsonSampling"]
            )

    def report(self) -> pd.DataFrame:
        """Save results to CSV and log summary.

        Args:

        Returns:
          pd.DataFrame: Experiment results.

        """
        df = pd.DataFrame(
            self.data,
            columns=[
                "Trial",
                "Bandit",
                "Reward",
                "CumulativeReward",
                "CumulativeRegret",
                "Algorithm",
            ],
        )
        df.to_csv("results_thompson.csv", index=False)
        avg_reward = df["Reward"].mean()
        total_regret = df["CumulativeRegret"].iloc[-1]
        logger.info(
            f"[Thompson Sampling] average reward: {avg_reward:.3f} | "
            f"Cumulative regret: {total_regret:.3f}"
        )
        return df


class Visualization:
    """Handles plotting of learning and performance metrics."""

    def __init__(self, results_df:pd.DataFrame) -> None:
        """Initialize with combined results DataFrame."""
        self.results = results_df

    def plot_learning_curves(self) -> None:
        """Visualize average reward evolution for each algorithm."""
        plt.figure(figsize=(12, 6))
        for algo in self.results["Algorithm"].unique():
            subset = (
                self.results[self.results["Algorithm"] == algo]
                .sort_values("Trial")
            )
            plt.plot(
                subset["Trial"],
                subset["Reward"].rolling(window=200).mean(),
                label=f"{algo} (smoothed)"
            )
        plt.title("Learning curves (smoothed average reward)")
        plt.xlabel("Trial")
        plt.ylabel("Average reward (200-trial rolling mean)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_metric(self, column:str, title:str, y_label:str) -> None:
        """
        General function to plot a metric for all algorithms.

        Args:
            column(str): Name of the column to plot.
            title(str): Plot title.
            y_label(str): Y-axis label.

        """
        plt.figure(figsize=(12, 5))
        for algo in self.results["Algorithm"].unique():
            subset = (
                self.results[self.results["Algorithm"] == algo]
                .sort_values("Trial")
            )
            plt.plot(subset["Trial"], subset[column], label=algo)
        plt.title(title)
        plt.xlabel("Trial")
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_comparisons(self) -> None:
        """Plot cumulative reward and regret using the general function."""
        self.plot_metric(
            column="CumulativeReward",
            title="Cumulative rewards comparison",
            y_label="Cumulative reward"
        )
        self.plot_metric(
            column="CumulativeRegret",
            title="Cumulative regret comparison",
            y_label="Cumulative regret"
        )


def comparison():
    """Run both algorithms and compare results visually and numerically."""
    bandit_reward = [1, 2, 3, 4]
    n_trials = 20000

    # Epsilon-Greedy
    logger.info("Running Epsilon-Greedy...")
    np.random.seed(1)
    eg = EpsilonGreedy(bandit_reward, n_trials)
    eg.experiment()
    df_eg = eg.report()

    # Thompson Sampling
    logger.info("Running Thompson Sampling...")
    np.random.seed(2)  # different seed for variety
    ts = ThompsonSampling(bandit_reward, n_trials)
    ts.experiment()
    df_ts = ts.report()

    # Combine results
    combined_df = pd.concat([df_eg, df_ts], ignore_index=True)
    combined_df.to_csv("bandit_results_comb.csv", index=False)

    # Visualization
    vis = Visualization(combined_df)
    vis.plot_learning_curves()
    vis.plot_comparisons()


if __name__ == "__main__":
    logger.info("Starting multi-armed bandit experiment comparison...")
    comparison()
