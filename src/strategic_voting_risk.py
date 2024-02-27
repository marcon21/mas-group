import numpy as np
import happiness_level as hl
import copy

def strategic_voting_risk1(strategies: np.array, pref: np.array):
    m, n = pref.shape
    m_fact = 1
    for i in range(m):
        m_fact = m_fact*(i+1)
    n_strategies = 0
    for si in strategies:
        n_strategies = n_strategies + si.shape[0]
    print(n_strategies)
    svr = n_strategies/(m_fact*n)
    return svr

def strategic_voting_risk2(strategies: np.array, pref: np.array):
    m, n = pref.shape
    n_strategic_voters = 0
    for si in strategies:
        if si.shape[0] != 0:
            n_strategic_voters+=1
    print(n_strategic_voters)
    svr = n_strategic_voters/n
    return svr

def strategies(pref: np.array) -> np.array:
    m, n = pref.shape
    new_voting = np.array([[pref[i][j] for i in range(m)] for j in range(n)])
    system = ''
    outcome = get_outcome(pref, system)
    for voter in new_voting:
        h_old = hl.ind_happiness(voter, outcome)
    return#NOT FINISHED

def get_outcome(pref, system):
    votes = get_votes(pref, system)
    outcome = ''
    return outcome

def combinations (options: np.array, s):
    s_all = list()
    for i in range(len(options)):
        s_copy = copy.copy(s)
        options_copy = copy.copy(options)
        s_copy.append(options_copy.pop(i))
        if len(options_copy) > 0:
            s_all.extend(combinations(options_copy, s_copy))
        else:
            s_all.append(s_copy)
    return s_all

def get_votes(pref, system):
    pass

def main():
    options  = ['A', 'B', 'c']
    s = []
    all_combinations = combinations(options, s)
    print(all_combinations)

if __name__ == '__main__':
    main()