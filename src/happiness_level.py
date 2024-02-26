import numpy as np
import math

def happiness_level(pref: np.array, outcome: str) -> np.array:
    m = pref.shape[0]
    n = pref.shape[1]
    new_voting = np.array([[pref[i][j] for i in range(m)] for j in range(n)])
    h = list()
    for ivoter in range(n):
        d = np.where(new_voting[ivoter]==outcome)[0][0]
        h.append(distr_h(d, m))
    return h

def distr_h(d: float, m: int) -> float:
    h_i = (1-2/(m-1)*d)
    k = 0.95
    c = 1/math.atanh(k)
    h = math.atanh(h_i*k)*c
    return h


