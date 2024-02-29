import numpy as np
from typing import Union
from src.outcomes import Result
import matplotlib.pyplot as plt
from src.utils import VotingArray
import math


class HappinessLevel:
    """
    Happiness level
    """

    def __init__(self, preferences: VotingArray, winner: Union[str, Result]) -> None:
        self.preferences = preferences
        self.columns = preferences.to_pandas().columns.to_list()
        self.voters_winner_rank = np.zeros(self.preferences.shape[1])

        if isinstance(winner, Result):
            self.winner = winner.winner
        else:
            self.winner = winner

    def __repr__(self) -> str:
        return f"Happiness level: {self.happiness_level}"

    @property
    def all_happiness_level(self) -> np.ndarray:
        """
        Returns the happiness level of all voters in a np.ndarray
        """
        num_voters = self.preferences.shape[1]
        happiness = np.zeros(num_voters)

        for i in range(num_voters):
            preference = self.preferences[:, i]  # Preferences of the i-th vote
            voter_winner_rank = np.where(preference == self.winner)[0][0]
            self.voters_winner_rank[i] = voter_winner_rank
            happiness[i] = self.happiness_level(voter_winner_rank)

        # Example of happiness: [0.5, 1, 1, 0]
        return happiness

    @property
    def sum_happiness_level(self) -> float:
        """
        Returns the sum of all happiness level
        """
        return self.all_happiness_level.sum()

    @property
    def happiness_level_dict(self) -> dict:
        """
        Returns the happiness level of all voters in a dictionary
        with the column names as keys and the happiness level as values
        """
        if self.columns is None:
            raise ValueError("Columns not defined")
        return dict(zip(self.columns, self.all_happiness_level))

    def happiness_level(self, vwr: int) -> float:
        """
        vwr -- (voter winner rank") preference rank of winner by voter i.
        """
        k = 0.95
        c = 1 / (2 * math.atanh(k))
        h_i = 1 - 2 / (self.preferences.shape[0] - 1) * vwr
        h = math.atanh(h_i * k) * c + 0.5
        return h

    def linear_happiness_level(self, vwr: int) -> float:
        """
        Simplified linear happiness level. Used as an estimate for stategical voting
        finder.

        vwr: (voter winner rank) preference rank of winner by voter i.
        """
        return 1 - np.divide(vwr, self.preferences.shape[0] - 1)

    def distribution_plot(self, show=False):
        x = np.linspace(0, self.preferences.shape[0] - 1, 1000)
        y = [self.happiness_level(v) for v in x]
        plt.plot(x, y)

        if show:
            plt.xlabel("Voter Winner Rank")
            plt.ylabel("Happiness Level")
            plt.title("Happiness Level Distribution")
            plt.grid(True)
            plt.show()

    def plot(self):
        """
        Graphs the happiness level of all voters ordered by their happiness level with matplotlib
        """
        self.distribution_plot()

        plt.scatter(self.voters_winner_rank, self.all_happiness_level, c="r")
        plt.xlabel("Voters Winner Rank")
        plt.ylabel("Happiness Level")
        plt.title("Happiness Level of All Voters")
        plt.grid(True)

        plt.show()

    def histogram(self):
        counts, bins = np.histogram(self.all_happiness_level)
        plt.hist(bins[:-1], bins, weights=counts)

        plt.xlabel("Happiness Level")
        plt.ylabel("Frequency")
        plt.title("Histogram of Happiness Level")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import utils
    from pprint import pprint
    import outcomes as o

    voting_array = utils.read_voting("input/voting_result.json", table_name="voting2")

    # voting_array = utils.random_voting(n_voters=100, n_candidates=50)
    winner = o.plurality_outcome(voting_array).winner

    print(voting_array.to_pandas(), f"\nWinner: {winner}", "\n")

    h = HappinessLevel(voting_array, winner)
    print(h.happiness_level_dict)

    # h.graph_happiness()
