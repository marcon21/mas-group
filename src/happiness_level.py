import numpy as np
from typing import Union
from src.outcomes import Result
import matplotlib.pyplot as plt
from src.utils import VotingArray
import math
from pandas import DataFrame
from src.utils import VotingSchemas


class HappinessLevel:
    """
    Happiness level
    """

    def __init__(
        self,
        preferences: VotingArray,
        winner: Union[str, Result],
        voting_schema: VotingSchemas,
    ) -> None:
        self.preferences = preferences
        self.columns = preferences.to_pandas().columns.to_list()
        self.voting_schema = voting_schema
        self.voters_winner_rank = np.zeros(self.preferences.shape[1])
        self._all_happiness_level = None

        if isinstance(winner, Result):
            self.winner = winner.winner
        else:
            self.winner = winner

    def run(self, show=False):
        """
        Runs the happiness level analysis.

        Args:
            display (bool, optional): Whether to display the analysis results. Defaults to False.

        Returns:
            self: The current instance of the HappinessLevel class.
        """
        _ = self.voter

        if show:
            print("Voters Happiness Level")
            display(self.happiness_level_pandas())  # Only for notebooks

            print(f"\nOverall Happiness Level: {self.total}")

            print("\nHappiness Level Distribution")
            self.plot()

            print("\nHistogram of Happiness Level")
            self.histogram()

        return self

    @property
    def voter(self) -> np.ndarray:
        """
        Returns the happiness level of all voters in a np.ndarray
        """
        if self._all_happiness_level is not None:
            return self._all_happiness_level

        num_voters = self.preferences.shape[1]
        self._all_happiness_level = np.zeros(num_voters)

        for i in range(num_voters):
            preference = self.preferences[:, i]  # Preferences of the i-th vote
            voter_winner_rank = np.where(preference == self.winner)[0][0]
            self.voters_winner_rank[i] = voter_winner_rank
            self._all_happiness_level[i] = self.happiness_level(voter_winner_rank)

        # Example of happiness: [0.5, 1, 1, 0]
        return self._all_happiness_level

    @property
    def total(self) -> float:
        """
        Returns the sum of all happiness level
        """
        return self.voter.sum()

    @property
    def happiness_level_dict(self) -> dict:
        """
        Returns the happiness level of all voters in a dictionary
        with the column names as keys and the happiness level as values
        """
        if self.columns is None:
            raise ValueError("Columns not defined")
        return dict(zip(self.columns, self.voter))

    def happiness_level_pandas(self) -> DataFrame:
        """
        Returns the happiness level of all voters in a pandas DataFrame
        """
        df = DataFrame(self.happiness_level_dict, index=[0]).T
        df.rename(columns={0: "Happiness Level"}, inplace=True)
        return df

    def happiness_level(self, vwr: int) -> float:
        """
        vwr -- (voter winner rank") preference rank of winner by voter i.
        """
        k = 0.95
        c = 1 / (2 * math.atanh(k))
        h_i = 1 - 2 / (self.preferences.shape[0] - 1) * vwr
        h = math.atanh(h_i * k) * c + 0.5
        return h

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

        plt.scatter(self.voters_winner_rank, self.voter, c="r")
        plt.xlabel("Voters Winner Rank")
        plt.ylabel("Happiness Level")
        plt.title(f"{self.voting_schema.value} -- Happiness Level of All Voters")
        plt.grid(True)

        plt.show()

    def histogram(self):
        counts, bins = np.histogram(self.voter)
        plt.hist(bins[:-1], bins, weights=counts)

        plt.xlabel("Happiness Level")
        plt.ylabel("Frequency")
        plt.title(f"{self.voting_schema.value} -- Histogram of Happiness Level")
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
