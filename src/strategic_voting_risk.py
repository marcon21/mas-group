from itertools import permutations
from typing import Callable, List, Tuple
from pandas import DataFrame
import numpy as np

from src.utils import VotingArray
from src.happiness_level import HappinessLevel


class StrategicVoting:
    def __init__(self, preferences: VotingArray) -> None:
        self.preferences = preferences
        self.all = DataFrame()

    def find(self, happiness: HappinessLevel, schema_outcome_f: Callable):
        result = schema_outcome_f(self.preferences)
        strategic_voting = []

        for i in range(self.preferences.shape[1]):
            vwr = np.argwhere(self.preferences[:, i] == result.winner)[0][0]

            for p in permutations(self.preferences[:, i]):
                p = np.array(p)
                # Consider only permutations where winner is placed below or in
                # the same position as the voter preference.
                if np.argwhere(p == result.winner)[0][0] >= vwr:
                    new_voting = self.preferences.copy()
                    new_voting[:, i] = p
                    new_result = schema_outcome_f(new_voting)
                    new_happiness = HappinessLevel(self.preferences, new_result.winner)
                    new_vwr = np.argwhere(self.preferences[:, i] == new_result.winner)[
                        0
                    ][0]
                    voter_happiness = new_happiness.linear_happiness_level(new_vwr)
                    if voter_happiness > happiness.voter[i]:
                        strategic_voting.append(
                            (
                                i,
                                p,
                                new_result,
                                new_happiness.voter[i],
                                happiness.voter[i],
                                new_happiness.total,
                                happiness.total,
                            )
                        )

        self._as_pandas(strategic_voting)

    def _as_pandas(self, strategic_voting: List[Tuple]) -> DataFrame:
        self.all = DataFrame(
            strategic_voting,
            columns=[
                "voter",
                "strategic_voting",
                "new_result",
                "strategic_H",
                "previous_H",
                "stategic_overall_H",
                "startegic_overall_H",
            ],
        )

        # Print only the winner from Result
        self.all["new_result"] = self.all["new_result"].apply(lambda x: x.winner)
