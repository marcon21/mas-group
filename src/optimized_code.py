import numpy as np
import math
from typing import Callable
from itertools import permutations
from collections import Counter
from joblib import Parallel, delayed
from numba import jit
import json 
from joblib import Parallel, delayed

def read_voting_(file_path: str, table_name: str = "voting"):
    with open(file_path) as f:
        voting = json.load(f)[table_name]
        voting = np.array(voting)
    return voting

def plurality_outcome_op(votes: np.ndarray) -> int:
    results = Counter(votes[0, :])
    max_count = max(results.values())
    return min(filter(lambda x: results[x] == max_count, results))

def for_two_outcome_op(arr: np.ndarray) -> int:
    counts = Counter(arr[:2].ravel()) 
    max_count = max(counts.values())
    max_elements = [key for key, value in counts.items() if value == max_count]
    return min(max_elements, default=None)

def veto_outcome_op(arr: np.ndarray) -> int:
    counts = Counter(arr[:-1].ravel())  # Count occurrences of elements
    max_count = max(counts.values(), default=0)  # Find the maximum count
    max_elements = [key for key, value in counts.items() if value == max_count]  # Find elements with maximum count
    return min(max_elements, default=None)  # Return the minimum element among those with maximum count

def borda_outcome_op(preferences: np.ndarray) -> int:
    n, m = preferences.shape
    alternatives, counts = np.unique(preferences, return_counts=True)

    # Initialize Borda points using vectorized operations
    borda_points = np.zeros(len(alternatives))
    for i, alternative in enumerate(alternatives):
        borda_points[i] = ((n - 1) * counts[i] - (np.where(preferences == alternative)[0] * (n - 1 - np.arange(n)).reshape(-1, 1)).sum())

    # Find the index of the maximum Borda points
    winner_index = np.argmax(borda_points)

    # Return the alternative corresponding to the winner index
    winner = alternatives[winner_index]
    return winner

def happiness_level(preferences: np.ndarray,num_voter: int, outcome: int) -> float:
    vwr = np.argmax(preferences[:, num_voter] == outcome)
    k = 0.95
    c = 1 / (2 * math.atanh(k))
    h_i = 1 - 2 / (preferences.shape[0] - 1) * vwr
    h = math.atanh(h_i * k) * c + 0.5
    return h

def happiness_level_total(preferences: np.ndarray, outcome: int) -> float:
    hap = np.zeros(preferences.shape[1])
    for voter in range(preferences.shape[1]):
        vwr = np.argmax(preferences[:, voter] == outcome)
        k = 0.95
        c = 1 / (2 * math.atanh(k))
        h_i = 1 - 2 / (preferences.shape[0] - 1) * vwr
        hap[voter] = math.atanh(h_i * k) * c + 0.5
    return hap


def compute_voter_risk(preferences: np.ndarray, result: int, initial_happinesses: np.ndarray, i: int, schema_outcome_f: Callable) -> float:
    """
    Compute the risk for a specific voter.

    Args:
        preferences (np.ndarray): The preferences matrix.
        result (int): The current outcome.
        initial_happinesses (np.ndarray): The initial happiness levels for all voters.
        i (int): Index of the voter for whom risk is calculated.
        schema_outcome_f (Callable): Function to calculate the outcome based on preferences.

    Returns:
        float: The risk for the specified voter.
    """
    initial_happiness = initial_happinesses[i]
    vwr_idx = np.flatnonzero(preferences[:, i] == result)
    if vwr_idx.size == 0:
        return 0  # Voter doesn't prefer the current outcome

    vwr = vwr_idx[0]
    best_happiness = initial_happiness

    perms = list(permutations(preferences[:, i]))
    perm_array = np.array(perms)

    new_voting = preferences.copy()
    for perm in perm_array:
        idx = np.argwhere(perm == result)[0][0]
        if idx >= vwr:
            new_voting[:, i] = perm
            new_result = schema_outcome_f(new_voting)
            new_happiness = happiness_level(preferences, i, new_result)
            best_happiness = max(best_happiness, new_happiness)

    return best_happiness - initial_happiness

def compute_risk(preferences: np.ndarray, schema_outcome_f: Callable) -> float:
    """
    Compute the average risk for all voters.

    Args:
        preferences (np.ndarray): The preferences matrix.
        schema_outcome_f (Callable): Function to calculate the outcome based on preferences.

    Returns:
        float: The average risk for all voters.
    """
    result = schema_outcome_f(preferences)
    initial_happinesses = happiness_level_total(preferences, result)
    num_unhappy_voters = np.count_nonzero(initial_happinesses != 1)

    if num_unhappy_voters == 0:
        return 0

    total_risk = sum(compute_voter_risk(preferences, result, initial_happinesses, voter, schema_outcome_f) 
                    for voter in range(preferences.shape[1]))

    return total_risk / num_unhappy_voters

def compute_voter_risk_parallel(preferences: np.ndarray, result: int, initial_happinesses: np.ndarray, i: int, schema_outcome_f: Callable) -> float:
    initial_happiness = initial_happinesses[i]
    vwr_idx = np.flatnonzero(preferences[:, i] == result)
    if vwr_idx.size == 0:
        return 0

    vwr = vwr_idx[0]
    best_happiness = initial_happiness

    perms = list(permutations(preferences[:, i]))
    perm_array = np.array(perms)

    new_voting = preferences.copy()
    for perm in perm_array:
        idx = np.argwhere(perm == result)[0][0]
        if idx >= vwr:
            new_voting[:, i] = perm
            new_result = schema_outcome_f(new_voting)
            new_happiness = happiness_level(preferences, i, new_result)
            best_happiness = max(best_happiness, new_happiness)

    return best_happiness - initial_happiness

def compute_risk_parallel(preferences: np.ndarray, schema_outcome_f: Callable,cores: int) -> float:
    result = schema_outcome_f(preferences)
    initial_happinesses = happiness_level_total(preferences, result)
    num_unhappy_voters = np.count_nonzero(initial_happinesses != 1)

    if num_unhappy_voters == 0:
        return 0

    num_voters = preferences.shape[1]
    risks = Parallel(n_jobs=cores)(delayed(compute_voter_risk_parallel)(preferences, result, initial_happinesses, voter, schema_outcome_f) for voter in range(num_voters))
    total_risk = sum(risks)

    return total_risk / num_unhappy_voters

def compute_voter_risk_combinations(preferences: np.ndarray, result: int, initial_happinesses: np.ndarray, voter_index: int, schema_outcome_f: Callable):
    valid_combinations = []
    initial_happiness = initial_happinesses[voter_index]
    valid_result_index = np.flatnonzero(preferences[:, voter_index] == result)
    if valid_result_index.size == 0:
        return 0, []

    valid_voting_result_index = valid_result_index[0]
    best_happiness = initial_happiness

    permutations_of_preferences = list(permutations(preferences[:, voter_index]))
    perm_array = np.array(permutations_of_preferences)

    new_voting = preferences.copy()
    for perm in perm_array:
        idx = np.argwhere(perm == result)[0][0]
        if idx >= valid_voting_result_index:
            new_voting[:, voter_index] = perm
            new_result = schema_outcome_f(new_voting)
            new_happiness = happiness_level(preferences, voter_index, new_result)  
            if new_happiness > best_happiness:
                valid_combinations.append(perm)
                best_happiness = new_happiness

    return best_happiness - initial_happiness, valid_combinations

def compute_risk_combinations(preferences: np.ndarray, schema_outcome_f: Callable):
    result = schema_outcome_f(preferences)
    valid_combinations_per_voter = [[] for _ in range(preferences.shape[1])]
    total_risk = 0
    initial_happinesses = happiness_level_total(preferences, result)
    final_happinesses = initial_happinesses.copy()
    num_unhappy_voters = np.count_nonzero(initial_happinesses != 1)

    if num_unhappy_voters == 0:
        return 0, valid_combinations_per_voter, final_happinesses

    for voter_index in range(preferences.shape[1]): 
        result_tuple = compute_voter_risk_combinations(preferences, result, initial_happinesses, voter_index, schema_outcome_f)
        if len(result_tuple) == 2:
            risk, valid_combinations = result_tuple
            new_happiness = initial_happinesses[voter_index]
        else:
            risk, valid_combinations, new_happiness = result_tuple

        total_risk += risk.sum()
        valid_combinations_per_voter[voter_index] = valid_combinations
        final_happinesses[voter_index] = new_happiness
        
    return total_risk / num_unhappy_voters, valid_combinations_per_voter, final_happinesses
