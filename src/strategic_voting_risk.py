import numpy as np

def strategic_voting_risk1(strategies: np.array, pref: np.array):
    m = pref.shape[0]
    n = pref.shape[1]
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
    n = pref.shape[1]
    n_strategic_voters = 0
    for si in strategies:
        if si.shape[0] != 0:
            n_strategic_voters+=1
    print(n_strategic_voters)
    svr = n_strategic_voters/n
    return svr
