import numpy as np
import pandas as pd
import math, json
from typing import Callable, Tuple, List
from itertools import permutations
from collections import Counter
import json 

    
def get_scheme(voting_scheme:np.ndarray)-> Callable:
    
    if all(voting_scheme[i] == len(voting_scheme) - i for i in range(len(voting_scheme))):
        return borda_outcome_op
    if voting_scheme[0] == 1 and voting_scheme[1] == 0:
        return plurality_outcome_op
        
    if voting_scheme[0] == 1 and voting_scheme[1] == 1  and voting_scheme[2] == 0:
        return for_two_outcome_op
        
    if all(voting_scheme[i] == 1 for i in range(len(voting_scheme) - 1)) and voting_scheme[-1] == 0:
        return veto_outcome_op
        
    raise ValueError("Invalid voting scheme")


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


def veto_outcome_op(arr: np.ndarray):
    counts = Counter(arr[:-1].ravel()) 
    max_count = max(counts.values())
    max_elements = [key for key, value in counts.items() if value == max_count]
    return min(max_elements)

def borda_outcome_op(preferences: np.ndarray) -> int:
    n, m = preferences.shape
    alternatives, counts = np.unique(preferences, return_counts=True)
    borda_points = np.zeros(len(alternatives))
    for i, alternative in enumerate(alternatives):
        borda_points[i] = ((n - 1) * counts[i] - (np.where(preferences == alternative)[0] * (n - 1 - np.arange(n)).reshape(-1, 1)).sum())
    winner_index = np.argmax(borda_points)
    winner = alternatives[winner_index]
    return winner

def happiness_level(vwr, candidates) -> float:
    k = 0.95
    c = 1 / (2 * math.atanh(k))
    h_i = 1 - 2 / (candidates - 1) * vwr
    h = math.atanh(h_i * k) * c + 0.5
    return h

def happiness_level_total(preferences: np.ndarray, outcome: int) -> np.ndarray:
    num_voters = preferences.shape[1]
    happiness_levels = np.zeros(num_voters)
    candidates = preferences.shape[0]
    for voter in range(num_voters):
        vwr = np.argwhere(preferences[:, voter] == outcome)[0][0]
        happiness_levels[voter] = happiness_level(vwr, candidates)
    return happiness_levels

def compute_voter_risk(preferences: np.ndarray, result: int, initial_happinesses: np.ndarray, i: int, schema_outcome_f: Callable) -> float:
    initial_happiness = initial_happinesses[i]
    candidates = preferences.shape[0]
    vwr = np.argwhere(preferences[:, i] == result)[0][0]
    best_happiness = initial_happiness

    for perm in permutations(preferences[:, i]):
        if np.argwhere(perm == result)[0][0] >= vwr:
            new_voting = preferences.copy()
            new_voting[:, i] = perm
            new_result = schema_outcome_f(new_voting)
            new_vwr = np.argwhere(preferences[:, i] == new_result)[0][0]
            new_happiness = happiness_level(new_vwr, candidates)
            best_happiness = max(best_happiness, new_happiness)

    return best_happiness - initial_happiness

def compute_risk(preferences: np.ndarray, schema_outcome_f: Callable) -> float:
    result = schema_outcome_f(preferences)
    initial_happinesses = happiness_level_total(preferences, result)
    num_unhappy_voters = np.count_nonzero(initial_happinesses != 1)

    total_risk = sum(compute_voter_risk(preferences, result, initial_happinesses, voter, schema_outcome_f) 
                    for voter in range(preferences.shape[1]))

    if num_unhappy_voters == 0:
        return 0
    return total_risk / num_unhappy_voters


def compute_voter_risk_combinations(preferences: np.ndarray, 
                                    result: int, 
                                    initial_happinesses: np.ndarray, 
                                    i: int, 
                                    schema_outcome_f: Callable,
                                    strategies_happiness: pd.DataFrame) -> float:
    initial_happiness = initial_happinesses[i]
    candidates = preferences.shape[0]
    vwr = np.argwhere(preferences[:, i] == result)[0][0]
    best_happiness = initial_happiness
    
    new_voting = preferences.copy()
    for perm in permutations(preferences[:, i]):
        if np.argwhere(perm == result)[0][0] >= vwr:
            new_voting[:, i] = perm
            new_result = schema_outcome_f(new_voting)
            new_vwr = np.argwhere(preferences[:, i] == new_result)[0][0]
            new_happiness = happiness_level(new_vwr, candidates)
            
            if new_happiness > initial_happiness:
                strategies_happiness.loc[len(strategies_happiness)] = [i, perm, new_result, new_happiness, initial_happiness, sum(happiness_level_total(preferences,new_result))] 
                
            best_happiness =  max(new_happiness,best_happiness)
    
    return best_happiness - initial_happiness

def compute_risk_combinations(preferences: np.ndarray, 
                              schema_outcome_f: Callable,
                              result=0
                              ) -> Tuple[float, pd.DataFrame]:
    
    result = schema_outcome_f(preferences)
    initial_happinesses = happiness_level_total(preferences, result)
    initial_overall_happiness = sum(initial_happinesses)
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