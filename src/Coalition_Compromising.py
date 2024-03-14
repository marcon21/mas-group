from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from src import utils
from src.outcomes import plurality_outcome
from src.happiness_level import HappinessLevel
import numpy as np 
import pandas as pd



def remove_elements_above_or_equal_index(lst, index): 
    if index < 0 or index >= len(lst):
        return lst  
    
    return lst[:index]  


def find_new_happiness(manipulations, coalition, voting_df): #Function that for a manipulation of a coalition gives you the new happiness values
    n_cand = len(voting_df.columns)-2
    coalition['manipulation']=manipulations
    indici = coalition.index
    df = pd.DataFrame(manipulations, index= indici, columns=[i for i in range(0, n_cand)])
    new_voting_df = voting_df.copy()
    new_voting_df.loc[indici] = df
    new_results = plurality_outcome(new_voting_df.iloc[:-2, :n_cand].values.T)
    diz = HappinessLevel(voting_df.iloc[:, :n_cand].values.T, new_results.winner).happiness_level_dict 
    New_Happiness_Levels = pd.DataFrame.from_dict(diz, orient='index', columns=['New_H']) #compute the happiness of the new results with respect of the real preferences
    coalition = pd.merge(coalition, New_Happiness_Levels,left_index=True, right_index=True)

    
    return coalition, new_results

def analyze_core(coalition): #Analize if inside or not the code
    

    real_happ = coalition['H']
    fake_happ = coalition['New_H']
    
    comparison_result = [fake > real for real, fake in zip(real_happ, fake_happ)]
    
    if all(comparison_result):
        return True
    else:
        return False
 
def find_stable_coalitions_by_compromising(max_coal, voting_df, happiness_level, results):
    
    win = results.winner
    voting_df['H']= happiness_level._all_happiness_level
    others = voting_df[voting_df[0]!=win]
    #Creating Dissimilarity Matrix
    rankings = np.array(others.T)
    cor_mat, _ = spearmanr(rankings[:-2])
    dsim_mat = np.around(1 - cor_mat, decimals=4)
    
    np.fill_diagonal(dsim_mat, 0)

    #find stable coalitions
    coal = []
    for num in range(max_coal, 1, -1):

        clustering = AgglomerativeClustering(n_clusters=num, affinity='precomputed', linkage='average') #do clustering
        clusters = clustering.fit_predict(dsim_mat)
        others['gruppo'] = clusters

        for coal_id, coalition in others.groupby('gruppo'):
        
        
            better_op=[]

            for index, row in coalition.iterrows():
                pref = list(row.iloc[0:-2])
                ind = pref.index(win) 
                candidates = set(remove_elements_above_or_equal_index(pref, ind)) #find the candidates above the winner
                better_op.append(candidates) 
        
            if len(better_op)>1: #if the coalition is bigg than 1
                intersection = better_op[1].copy()
                for el in better_op:
                    intersection = set.intersection(el, intersection)

                if len(intersection)>0:

                    for alt in intersection: #try the alternatives in the intersection

                        man=[] #initialize the list with the voters manipulations 

                        for index, row in coalition.iterrows():
                            pref = list(row.iloc[0:-2])
                            ind = pref.index(alt)
                            pref.pop(ind)
                            pref.insert(0, alt)
                            man.append(pref) 

                        coal_new_h, new_result = find_new_happiness(man, coalition, voting_df) #compute the new happiness

                        coalition = coalition.iloc[:, :-1]

                        if analyze_core(coal_new_h) == True:

                            print(f'Burying {alt} made everyone in the group {coal_id} happier, here the new winner:  ', new_result)
                            coal.append((coal_new_h, new_result))

    return coal

                    

                

