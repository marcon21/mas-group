import numpy as np
import math
from typing import Callable
from itertools import permutations
from collections import Counter
from joblib import Parallel, delayed
from typing import Callable, List

def calculate_voter_risk_and_happinesses(preferences, result, initial_happiness, i, schema_outcome_f):
    vwr_index = np.argmax(preferences[:, i] == result)
    best_happiness = initial_happiness[i]
    increasing_permutations = []
    for p in permutations(preferences[:, i]):
        p = np.array(p)
        p_winner_index = np.argmax(p == result)
        if p_winner_index >= vwr_index:
            new_result = schema_outcome_f(preferences[:, i] if p_winner_index == vwr_index else p)
            new_happiness = happiness_level(preferences if p_winner_index == vwr_index else np.column_stack((preferences[:, :i], p, preferences[:, i+1:])), i, new_result)
            if new_happiness > best_happiness:
                best_happiness = new_happiness
                increasing_permutations = [p]
            elif new_happiness == best_happiness:
                increasing_permutations.append(p)
    return best_happiness - initial_happiness[i], increasing_permutations

def calculate_risk_and_happinesses(preferences: np.ndarray, schema_outcome_f: Callable) -> (float, List[np.ndarray]):
    result = schema_outcome_f(preferences)
    initial_happiness = np.array([happiness_level(preferences, voter, result) for voter in range(preferences.shape[1])])
    total_risk, all_increasing_permutations = zip(*Parallel(n_jobs=-1)(
        delayed(calculate_voter_risk_and_happinesses)(preferences, result, initial_happiness, i, schema_outcome_f) for i in range(preferences.shape[1])))
    num_unhappy_voters = np.count_nonzero(initial_happiness != 1)
    avg_risk = sum(total_risk) / num_unhappy_voters if num_unhappy_voters != 0 else 0
    return avg_risk, [perm for perm_list in all_increasing_permutations for perm in perm_list]


def generate_random_voting_situation_np(num_voters: int, num_candidates: int) -> np.ndarray:
    return np.array([np.random.permutation(np.arange(num_candidates)) for _ in range(num_voters)]).T

def plurality_outcome(votes):
    results = Counter(votes[0, :])
    max_count = max(results.values())
    return min(filter(lambda x: results[x] == max_count, results))

def for_two_outcome(votes):
    results = Counter(np.concatenate((votes[0, :], votes[1, :])))
    max_count = max(results.values())
    return min(filter(lambda x: results[x] == max_count, results))

def veto_outcome(votes):
    results = Counter(votes[:-1, :].ravel())
    max_count = max(results.values())
    return min(filter(lambda x: results[x] == max_count, results))

def borda_outcome(preferences: np.ndarray) -> int:
    borda_points = Counter()
    for voter_preferences in preferences.T:
        borda_points.update({preference: (preferences.shape[0] - i - 1) for i, preference in enumerate(voter_preferences)})
    max_points = max(borda_points.values())
    max_alternatives = [alt for alt, points in borda_points.items() if points == max_points]
    return min(max_alternatives)

def happiness_level(preferences: np.ndarray, voter: int, outcome: int) -> float:
    k = 0.95
    c = 1 / (2 * math.atanh(k))
    vwr_idx = np.where(preferences[:, voter] == outcome)[0][0]
    h_i = 1 - 2 / (preferences.shape[0] - 1) * vwr_idx
    return math.atanh(h_i * k) * c + 0.5

def compute_voter_risk(preferences, result, initial_happinesses, i, schema_outcome_f):
    vwr_idx = np.flatnonzero(preferences[:, i] == result)[0]
    best_happiness = initial_happinesses[i]
    for p in permutations(preferences[:, i]):
        p = np.array(p)
        p_winner_idx = np.flatnonzero(p == result)[0]
        if p_winner_idx >= vwr_idx:
            new_voting = preferences.copy()
            new_voting[:, i] = p
            new_result = schema_outcome_f(new_voting)
            new_happiness = happiness_level(new_voting, i, new_result)
            best_happiness = max(best_happiness, new_happiness)
    return best_happiness - initial_happinesses[i]

def compute_risk(preferences: np.ndarray, schema_outcome_f: Callable) -> float:
    result = schema_outcome_f(preferences)
    initial_happinesses = np.array([happiness_level(preferences, voter, result) for voter in range(preferences.shape[1])])
    total_risk = sum(Parallel(n_jobs=-1)(delayed(compute_voter_risk)(preferences, result, initial_happinesses, i, schema_outcome_f) for i in range(preferences.shape[1])))
    num_unhappy_voters = np.count_nonzero(initial_happinesses != 1)
    return total_risk / num_unhappy_voters if num_unhappy_voters != 0 else 0