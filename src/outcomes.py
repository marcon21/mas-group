from typing import List, Tuple
import numpy as np


def plurality_outcome(s: np.array) -> List[Tuple[str, int]]:
    first_prefs = s[0, :]
    alternatives, votes = np.unique(first_prefs, return_counts=True)
    winners_i = np.argwhere(votes == np.max(votes)).flatten()
    winners = [(alternatives[i], np.max(votes)) for i in winners_i]
    winners = tie_votes_resolver(winners)
    return winners


def for_two_outcome(s: np.array) -> List[Tuple[str, int]]:
    return [()]


def veto_outcome(s: np.array) -> str:
    return ""


def borda_outcome(s: np.array) -> str:

    return ""


def tie_votes_resolver(r: np.array):
    return r


def all_schemas_outcomes(s: np.array) -> dict:
    scheme_func = {
        "Plurality Voting": plurality_outcome,
        "Voting for Two": for_two_outcome,
        "Veto Voting": veto_outcome,
        "Borda Voting": borda_outcome,
    }
    scheme_outcome = {n: f(s) for n, f in scheme_func.items()}
    return scheme_outcome
