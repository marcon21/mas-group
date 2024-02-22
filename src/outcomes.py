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
    first_prefs = s[:2, :]
    alternatives, votes = np.unique(first_prefs, return_counts=True)
    winners_i = np.argwhere(votes == np.max(votes)).flatten()
    winners = [(alternatives[i], np.max(votes)) for i in winners_i]
    winners = tie_votes_resolver(winners, n=2)
    return winners


def veto_outcome(s: np.array) -> str:
    return ""


def borda_outcome(s: np.array) -> str:

    return ""


def tie_votes_resolver(r: np.array, n: int = 1):
    """
    r: voting result
    n: num winners
    """
    if len(r) == n:
        return r

    # Sort result alphabetically and get first n winners
    sorted_r = np.sort([w[0] for w in r])[:n]
    r = [(w, r[0][1]) for w in sorted_r]
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
