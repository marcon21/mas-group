import numpy as np
import pandas as pd
from typing import List, Union
from outcomes import Result


# Happiness level, ranges from 0 to -inf
class HappinessLevel:
    def __init__(
        self, preferences: Union[np.ndarray, pd.DataFrame], winner: Union[str, Result]
    ) -> None:
        if isinstance(preferences, pd.DataFrame):
            # If preferences is a pd.DataFrame
            self.preferences = preferences.to_numpy()
            self.columns = preferences.columns.to_list()
        elif isinstance(preferences, pd.Series):
            # If preferences is a pd.Series
            self.preferences = np.array([[x] for x in preferences.to_numpy()])
            self.columns = [preferences.name]
        else:
            # If preferences is a np.ndarray
            self.preferences = preferences
            self.columns = None

        if isinstance(winner, Result):
            self.winner = winner.winner
        else:
            self.winner = winner

    @property
    def all_happiness_level(self) -> np.ndarray:
        happiness = np.zeros(self.preferences.shape[1])

        # For each voter, find the index of the winner and assign the happiness level
        for i in range(self.preferences.shape[1]):
            personal_preferences = self.preferences[:, i]
            winner_index = np.where(personal_preferences == winner)[0][0]
            happiness[i] = -winner_index

        return happiness

    @property
    def happiness_level(self) -> float:
        return self.all_happiness_level.mean()

    @property
    def happiness_level_dict(self) -> dict:
        if self.columns is None:
            raise ValueError("Columns not defined")
        return dict(zip(self.columns, self.all_happiness_level))

    def __repr__(self) -> str:
        return f"Happiness level: {self.happiness_level}"


if __name__ == "__main__":
    import utils
    from pprint import pprint
    import outcomes as o

    voting_table = utils.read_voting(
        "../input/voting_result.json", table_name="voting2"
    ).to_pandas()
    winner = o.plurality_outcome(voting_table).winner

    print(voting_table, f"\nWinner: {winner}", "\n")

    h = HappinessLevel(voting_table, winner)
    print(h.happiness_level_dict)
    # print(voting_table.columns.to_list(), h)

    h = HappinessLevel(voting_table.iloc[:, 0], winner)
    print(h.happiness_level_dict)
