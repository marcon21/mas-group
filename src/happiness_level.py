import numpy as np
import pandas as pd
from typing import List, Union
from outcomes import Result
from utils import VotingArray
import math


def happiness_level(pref: np.array, outcome: str) -> np.array:
    m = pref.shape[0]
    n = pref.shape[1]
    new_voting = np.array([[pref[i][j] for i in range(m)] for j in range(n)])
    h = list()
    for ivoter in range(n):
        d = np.where(new_voting[ivoter] == outcome)[0][0]
        h.append(distr_h(d, m))
    return h


def distr_h(d: float, m: int) -> float:
    h_i = 1 - 2 / (m - 1) * d
    k = 0.95
    c = 1 / math.atanh(k)
    h = math.atanh(h_i * k) * c
    return h


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
            preference = self.preferences[:, i]  # Preferences of the i-th voter
            # happiness[i] = HAPPINESS_LEVEL
            # self.winner for accessing the winner : str
            # YOUR CODE HERE FOR CALCULATING THE HAPPINESS LEVEL

        # YOUR CODE HERE FOR MAP THE VALUES

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

    def __repr__(self) -> str:
        return f"Happiness level: {self.happiness_level}"


if __name__ == "__main__":
    import utils
    from pprint import pprint
    import outcomes as o

    voting_array = utils.read_voting("../input/voting_result.json", table_name="voting")
    winner = o.plurality_outcome(voting_array).winner

    print(voting_array.to_pandas(), f"\nWinner: {winner}", "\n")

    h = HappinessLevel(voting_array, winner)
    print(h.happiness_level_dict)
