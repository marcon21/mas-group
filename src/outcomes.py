from typing import List, Tuple
import numpy as np


def plurality_outcome(s: np.array) -> List[Tuple[str, int]]:
    return ""


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
