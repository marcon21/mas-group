import numpy as np
import math
from typing import Callable, Tuple, List
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

def compute_risk(preferences: np.ndarray, schema_outcome_f: Callable) -> float:
    result = schema_outcome_f(preferences)
    initial_happinesses = happiness_level_total(preferences, result)
    num_unhappy_voters = np.count_nonzero(initial_happinesses != 1)

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

def compute_voter_risk_combinations(preferences: np.ndarray, 
                                    result: int, 
                                    initial_happinesses: np.ndarray, 
                                    i: int, 
                                    schema_outcome_f: Callable,
                                    strategies_happiness: pd.DataFrame) -> float:
    initial_happiness = initial_happinesses[i]
    vwr_idx = np.flatnonzero(preferences[:, i] == result)
    
    if not vwr_idx.size:
        return 0
    
    vwr = vwr_idx[0]
    best_happiness = initial_happiness
    
    for perm in permutations(preferences[:, i]):
        idx = np.where(perm == result)[0][0]
        
        if idx >= vwr:
            new_voting = preferences.copy()
            new_voting[:, i] = perm
            new_result = schema_outcome_f(new_voting)
            new_happiness = happiness_level(preferences, i, new_result)
            
            if new_happiness > initial_happiness:
                strategies_happiness.loc[len(strategies_happiness)] = [i, perm, new_result, new_happiness, initial_happiness, 0]  # Skip unnecessary computation of new_overall_happiness
                
            if new_happiness == 1:
                return 1 - initial_happiness
            
            best_happiness = max(best_happiness, new_happiness)
    
    return best_happiness - initial_happiness

def compute_risk_combinations(preferences: np.ndarray, 
                              schema_outcome_f: Callable) -> Tuple[float, pd.DataFrame]:
    
    result = schema_outcome_f(preferences)
    initial_happinesses = happiness_level_total(preferences, result)
    initial_overall_happiness = initial_happinesses.sum()
    num_unhappy_voters = np.count_nonzero(initial_happinesses != 1)
    strategies_happiness = pd.DataFrame(columns=["voter", "combination", "new_result", "strategic_happiness", "old_happiness", "overall_happiness"])

    total_risk = 0
    
    for voter in range(preferences.shape[1]):
        risk= compute_voter_risk_combinations(preferences, result, initial_happinesses, voter, schema_outcome_f,strategies_happiness) 
        total_risk += risk
    
    if total_risk == 0:
        return 0, strategies_happiness
    strategies_happiness["initial overall happiness"] = initial_overall_happiness
    return total_risk / num_unhappy_voters, strategies_happiness