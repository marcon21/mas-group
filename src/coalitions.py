from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from src import utils
from src.outcomes import plurality_outcome
from src.happiness_level import HappinessLevel
import numpy as np
import pandas as pd
import hashlib


def get_df_hash(df):
    df_bytes = df.to_json().encode()
    return hashlib.md5(df_bytes).hexdigest()


def remove_elements_above_or_equal_index(lst, index):
    if index < 0 or index >= len(lst):
        return lst

    return lst[:index]


def find_new_happiness(
    manipulations, coalition, voting_df, voting_schema
):  # Function that for a manipulation of a coalition gives you the new happiness values
    n_cand = len(voting_df.columns) - 1
    coalition["manipulation"] = manipulations
    indici = coalition.index
    df = pd.DataFrame(
        manipulations, index=indici, columns=[i for i in range(0, n_cand)]
    )
    new_voting_df = voting_df.copy()
    new_voting_df.loc[indici] = df
    new_results = plurality_outcome(new_voting_df.iloc[:, :n_cand].values.T)
    diz = HappinessLevel(
        voting_df.iloc[:, :n_cand].values.T, new_results.winner, voting_schema
    ).run()
    New_Happiness_Levels = pd.DataFrame.from_dict(
        diz.happiness_level_dict, orient="index", columns=["New_H"]
    )  # compute the happiness of the new results with respect of the real preferences
    New_Happiness_Levels["strategic_overall_H"] = diz.total
    coalition = pd.merge(
        coalition, New_Happiness_Levels, left_index=True, right_index=True
    )

    return coalition, new_results


def analyze_core(coalition):  # Analize if inside or not the code

    real_happ = coalition["H"]
    fake_happ = coalition["New_H"]

    comparison_result = [fake > real for real, fake in zip(real_happ, fake_happ)]

    if all(comparison_result):
        return True
    else:
        return False


def build_coalition_table(coalitions, previous_H):
    columns = [
        "coalition_group",
        "voter",
        "strategic_voting",
        "new_result",
        "strategic_H",
        "previous_H",
        "strategic_overall_H",
        "previous_overall_H",
    ]

    coalitions_list = []
    for c in coalitions:
        c[0].rename(
            {
                "H": "previous_H",
                "New_H": "strategic_H",
                "manipulation": "strategic_voting",
                "gruppo": "coalition_group",
            },
            axis=1,
            inplace=True,
        )
        c[0]["voter"] = c[0].index
        c[0]["new_result"] = c[1].winner
        coalitions_list.append(c[0])

    if coalitions_list:
        coalitions_df = pd.concat(coalitions_list)
        coalitions_df["previous_overall_H"] = previous_H
        coalitions_df = coalitions_df[columns]
        coalitions_df.reset_index(drop=True, inplace=True)
        return coalitions_df

    return pd.DataFrame(columns=columns)


def find_stable_coalitions_by_compromising(
    max_coal, voting_df, happiness_level, results, voting_schema
):

    win = results.winner
    voting_df["H"] = happiness_level._all_happiness_level
    others = voting_df[voting_df[0] != win]
    # Creating Dissimilarity Matrix
    rankings = np.array(others.T)
    cor_mat, _ = spearmanr(rankings[:-2])
    dsim_mat = np.around(1 - cor_mat, decimals=4)
    np.fill_diagonal(dsim_mat, 0)

    # find stable coalitions
    coal = []
    coal_index = {}
    cluster_epoch = 0
    for num in range(max_coal, 1, -1):

        # TODO: why average? and not ward?
        clustering = AgglomerativeClustering(
            n_clusters=num, metric="precomputed", linkage="average"
        )  # do clustering
        clusters = clustering.fit_predict(dsim_mat) + cluster_epoch
        cluster_epoch += max_coal + 1
        others["gruppo"] = clusters

        for coal_id, coalition in others.groupby("gruppo"):

            if get_df_hash(coalition.iloc[:, :-2]) not in coal_index:
                better_op = []
                coal_index[get_df_hash(coalition.iloc[:, :-2])] = coalition

                for index, row in coalition.iterrows():
                    pref = list(row.iloc[0:-2])
                    ind = pref.index(win)
                    candidates = set(
                        remove_elements_above_or_equal_index(pref, ind)
                    )  # find the candidates above the winner
                    better_op.append(candidates)

                if len(better_op) > 1:  # if the coalition is bigg than 1
                    intersection = better_op[1].copy()
                    for el in better_op:
                        intersection = set.intersection(el, intersection)

                    if len(intersection) > 0:

                        for (
                            alt
                        ) in intersection:  # try the alternatives in the intersection

                            man = (
                                []
                            )  # initialize the list with the voters manipulations

                            for index, row in coalition.iterrows():
                                pref = list(row.iloc[0:-2])
                                ind = pref.index(alt)
                                pref.pop(ind)
                                pref.insert(0, alt)
                                man.append(pref)
                            coal_new_h, new_result = find_new_happiness(
                                man, coalition, voting_df, voting_schema
                            )  # compute the new happiness

                            coalition = coalition.iloc[:, :-1]

                            if analyze_core(coal_new_h) == True:

                                print(
                                    f"Pushing {alt} made everyone in the group {coal_id} happier, here the new winner:  ",
                                    new_result,
                                )
                                coal.append((coal_new_h, new_result))

    return build_coalition_table(coal, happiness_level.total)