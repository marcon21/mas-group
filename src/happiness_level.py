import numpy as np
import pandas as pd
from typing import List, Union
from outcomes import Result
from utils import VotingArray
import math


class HappinessLevel:
    """
    Happiness level
    """

    def __init__(self, preferences: VotingArray, winner: Union[str, Result]) -> None:
        self.preferences = preferences
        self.columns = preferences.to_pandas().columns.to_list()

        if isinstance(winner, Result):
            self.winner = winner.winner
        else:
            self.winner = winner

    @property
    def all_happiness_level(self) -> np.ndarray:
        """
        Returns the happiness level of all voters in a np.ndarray
        """
        happiness = np.zeros(self.preferences.shape[1])

        for i in range(self.preferences.shape[1]):
            preference = self.preferences[:, i]  # Preferences of the i-th vote
            happiness[i] = np.where(preference == self.winner)[0][0]

        happiness = np.array([self.distr_h(h) for h in happiness])

        # Example of happiness: [0.5, 1, 1, 0]
        return happiness

    @property
    def happiness_level(self) -> float:
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

    def distr_h(self, h: float) -> float:
        k = 0.95
        c = 1 / (2 * math.atanh(k))
        h_i = 1 - 2 / (self.preferences.shape[0] - 1) * h
        h = math.atanh(h_i * k) * c + 0.5
        return h

    def __repr__(self) -> str:
        return f"Happiness level: {self.happiness_level}"

    def graph_happiness(self):
        """
        Graphs the happiness level of all voters ordered by their happiness level with matplotlib
        """
        import matplotlib.pyplot as plt

        happiness = self.happiness_level_dict
        happiness = dict(sorted(happiness.items(), key=lambda item: item[1]))

        plt.plot(list(happiness.values()), "o-")
        plt.xlabel("Voter")
        plt.xticks(range(len(happiness)), list(happiness.keys()))
        plt.xticks(rotation=45)
        plt.ylabel("Happiness level")
        plt.title("Happiness level of all voters")
        plt.show()


if __name__ == "__main__":
    import utils
    from pprint import pprint
    import outcomes as o

    voting_array = utils.read_voting(
        "../input/voting_result.json", table_name="voting2"
    )

    # voting_array = utils.random_voting(n_voters=100, n_candidates=50)
    winner = o.plurality_outcome(voting_array).winner

    print(voting_array.to_pandas(), f"\nWinner: {winner}", "\n")

    h = HappinessLevel(voting_array, winner)
    print(h.happiness_level_dict)

    # h.graph_happiness()
