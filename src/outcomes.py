from typing import List, Tuple, Union
import numpy as np
import pandas as pd

UnionArray = Union[np.ndarray, pd.DataFrame]


# Class to store the results of the voting,
# like a dictionary but with a winner field
class Result(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Method to get the winner of the voting, resolving ties by
    # returning the first one in alphabetical order
    @property
    def winner(self) -> str:
        return max(self, key=self.get)

    # Overriding the __repr__ method to include the winner
    def __repr__(self):
        return f"Winner: {self.winner}\n{super().__repr__()}"


# Decorator to wrap the outcome functions, to accept both numpy arrays and
# pandas dataframes and return a Result object
def outcome_wrapper(func):
    def wrapper(preferences: UnionArray, *args, **kwargs):
        if isinstance(preferences, pd.DataFrame):
            s = preferences.to_numpy()
        else:
            s = preferences

        res = func(s, *args, **kwargs)

        if isinstance(res, dict):
            all_alternatives = np.unique(preferences)
            result_dict = {key: res.get(key, 0) for key in all_alternatives}
            return Result(result_dict)

        return res

    return wrapper


# Voting for one
@outcome_wrapper
def plurality_outcome(preferences: UnionArray) -> Result:
    first_prefs = preferences[0, :]
    alternatives, votes = np.unique(first_prefs, return_counts=True)
    results = dict(zip(alternatives, votes))

    return results


# Voting for two
@outcome_wrapper
def for_two_outcome(preferences: UnionArray) -> Result:
    first_prefs = preferences[0, :]
    second_prefs = preferences[1, :]

    first_votes = dict(zip(*np.unique(first_prefs, return_counts=True)))
    second_votes = dict(zip(*np.unique(second_prefs, return_counts=True)))

    results = {
        k: first_votes.get(k, 0) + second_votes.get(k, 0)
        for k in first_votes.keys() | second_votes.keys()
    }

    return results


# Voting for veto
@outcome_wrapper
def veto_outcome(preferences: UnionArray) -> Result:
    last_prefs = preferences[:-1, :]
    alternatives, votes = np.unique(last_prefs, return_counts=True)
    results = dict(zip(alternatives, votes))

    return results


# Borda voting
@outcome_wrapper
def borda_outcome(preferences: UnionArray) -> Result:
    n, m = preferences.shape
    alternatives = np.unique(preferences)
    borda_points = {a: 0 for a in alternatives}

    for i in range(n):
        row = preferences[i, :]
        alternatives, votes = np.unique(row, return_counts=True)
        for a in alternatives:
            borda_points[a] += (n - 1 - i) * votes[np.where(alternatives == a)][0]

    return borda_points


# Getting the outcomes for all the voting schemas
def all_schemas_outcomes(s: UnionArray) -> Result:
    scheme_func = {
        "Plurality Voting": plurality_outcome,
        "Voting for Two": for_two_outcome,
        "Veto Voting": veto_outcome,
        "Borda Voting": borda_outcome,
    }

    scheme_outcome = {n: f(s) for n, f in scheme_func.items()}
    return scheme_outcome


if __name__ == "__main__":
    import utils
    from pprint import pprint

    voting_table = utils.read_voting(
        "../input/voting_result.json", table_name="voting2"
    ).to_pandas()

    print(voting_table, "\n")

    outcomes = all_schemas_outcomes(voting_table)
    for v, o in outcomes.items():
        print(f"{v}:\n{o}\n")

    # outcome = plurality_outcome(voting_table)
    # print(outcome)
