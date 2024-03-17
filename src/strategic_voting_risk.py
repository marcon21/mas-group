from itertools import permutations
from typing import Callable, List, Tuple
from pandas import DataFrame
import numpy as np

from src.utils import VotingArray
from src.happiness_level import HappinessLevel


class StrategicVoting:
    def __init__(self, preferences: VotingArray):
        self.preferences: VotingArray = preferences
        self.happiness: HappinessLevel = None

        self.all = DataFrame()
        self.best = DataFrame()
        self.risk = None

    def run(self, happiness: HappinessLevel, schema_outcome_f: Callable):
        self.happiness = happiness
        self._find_all(schema_outcome_f)
        self._find_best()
        self._compute_risk()
        return self

    def _find_all(self, schema_outcome_f: Callable):
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
                    if voter_happiness > self.happiness.voter[i]:
                        strategic_voting.append(
                            (
                                i,
                                p,
                                new_result,
                                new_happiness.voter[i],
                                self.happiness.voter[i],
                                new_happiness.total,
                                self.happiness.total,
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
                "strategic_overall_H",
                "previous_overall_H",
            ],
        )

        # Print only the winner from Result
        self.all["new_result"] = self.all["new_result"].apply(lambda x: x.winner)

    def _find_best(self):
        """
        Build table with best strategic voting for each voter. The best strat is defined
        as the strat overall voter strats with max strategic_H and strategic_overall_H.

        Sorts all strat votign table by decreasing "strategic_H", "strategic_overall_H".
        Drops voter duplicates by keeping the first instance in the table (hence the one)
        with highest ("strategic_H", "strategic_overall_H").
        """
        if self.all.empty:
            AttributeError("All strategic voting missing. Call find()")

        self.best = self.all.sort_values(
            ["strategic_H", "strategic_overall_H"], ascending=False
        ).drop_duplicates(subset=["voter"], keep="first")

    def _compute_risk(self):
        """
        First naive implementation of risk.
        """
        if self.best.empty:
            AttributeError("Best strategic voting missing. Call find_best()")

        num_unhappy_voters = np.where(self.happiness.voter != 1)[0].shape[0]
        voter_risk = (self.best["strategic_H"] - self.best["previous_H"]).sum()
        self.risk = voter_risk / num_unhappy_voters
