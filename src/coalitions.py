import numpy as np
import pandas as pd
import hashlib
from itertools import combinations
from functools import lru_cache
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from src import utils
from src.outcomes import plurality_outcome
from src.outcomes import borda_outcome
from src.happiness_level import HappinessLevel
import functools
import pickle


def memoize(func):
    cache = {}

    @functools.wraps(func)
    def memoized_func(*args):
        hash_value = hashlib.sha1(pickle.dumps(args)).hexdigest()
        if hash_value not in cache:
            cache[hash_value] = func(*args)
        return cache[hash_value]

    return memoized_func


def get_df_hash(df):
    df_bytes = df.to_json().encode()
    return hashlib.md5(df_bytes).hexdigest()


def remove_elements_above_or_equal_index(lst, index):
    if index < 0 or index >= len(lst):
        return lst

    return lst[:index]


@memoize
def compromise(new_poss_coal, results):
    better_op = []

    for index, row in new_poss_coal.iterrows():
        pref = list(row.iloc[0:-4])
        ind = pref.index(results.winner)
        candidates = set(
            remove_elements_above_or_equal_index(pref, ind)
        )  # find the candidates above the winner
        better_op.append(candidates)

    if len(better_op) > 1:  # if the coalition is bigg than 1
        intersection = better_op[1].copy()
        for el in better_op:
            intersection = set.intersection(el, intersection)

        if len(intersection) > 0:

            for alt in intersection:  # try the alternatives in the intersection

                man = []  # initialize the list with the voters manipulations

                for index, row in new_poss_coal.iterrows():
                    pref = list(row.iloc[0:-4])
                    ind = pref.index(alt)
                    pref.pop(ind)
                    pref.insert(0, alt)
                    man.append(pref)
    return man


def find_new_happiness(
    manipulations, coalition, voting_df, voting_schema, voting_schema_f
):  # Function that for a manipulation of a coalition gives you the new happiness values
    n_cand = len(voting_df.columns) - 1
    coalition["manipulation"] = manipulations
    indici = coalition.index
    df = pd.DataFrame(
        manipulations, index=indici, columns=[i for i in range(0, n_cand)]
    )

    new_voting_df = voting_df.copy()
    new_voting_df.loc[indici] = df
    new_results = voting_schema_f(new_voting_df.iloc[:, :n_cand].values.T)

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


def find_new_happiness2(
    manipulations, coalition, voting_df, voting_schema, voting_schema_f
):  # Function that for a manipulation of a coalition gives you the new happiness values
    n_cand = len(voting_df.columns) - 1
    coalition["manipulation"] = manipulations
    indici = coalition.index
    df = pd.DataFrame(
        manipulations, index=indici, columns=[i for i in range(0, n_cand + 1)]
    )
    new_voting_df = voting_df.copy()

    new_voting_df.loc[indici] = df
    new_results = voting_schema_f(new_voting_df.iloc[:, :n_cand].values.T)

    diz = HappinessLevel(
        voting_df.iloc[:, :n_cand].values.T, new_results.winner, voting_schema
    ).run()
    New_Happiness_Levels = pd.DataFrame.from_dict(
        diz.happiness_level_dict, orient="index", columns=["New_H_subcoal"]
    )  # compute the happiness of the new results with respect of the real preferences
    coalition = pd.merge(
        coalition, New_Happiness_Levels, left_index=True, right_index=True
    )

    return coalition, new_results


def analyze_core(coalition, var1, var2):  # Analize if inside or not the code
    coalition = coalition.loc[:, ~coalition.columns.duplicated()].copy()
    real_happ = coalition[var1]
    fake_happ = coalition[var2]

    comparison_result = [fake > real for real, fake in zip(real_happ, fake_happ)]

    if all(comparison_result):
        return True
    else:
        return False


def stability_of_coalitions(coal, voting_df, results, voting_schema, voting_schema_f):
    combinations_list = []
    stable = True
    r = 2
    max_r = len(coal.index)
    subcoalitions = []
    while r < max_r:
        combinations_list.extend(
            list(combinations(coal.index, r))
        )  # you check in the coalition, because they are theoretically the most similar.
        for el in combinations_list:
            indice = pd.Index(el)
            new_poss_coal = coal.loc[indice]
            man = compromise(new_poss_coal, results)
            coal_new_h, new_result = find_new_happiness2(
                man,
                new_poss_coal.iloc[:, :-1],
                voting_df,
                voting_schema,
                voting_schema_f,
            )  # compute the new happiness

            if analyze_core(coal_new_h, "H", "New_H_subcoal") == True:

                is_stable = pd.concat([coal_new_h, new_poss_coal["New_H"]], axis=1)

                if analyze_core(is_stable, "New_H", "New_H_subcoal") == True:
                    stable = False
                    subcoalitions.append(is_stable)

        combinations_list = []
        r += 1
    if len(subcoalitions) > 0:
        return stable, subcoalitions
    else:

        return stable, coal


def find_stable_coalitions_by_compromising(
    max_coal, voting_df, happiness_level, results, voting_schema, voting_schema_f
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
    scoal_index = {}
    cluster_epoch = 0
    for num in range(max_coal, 1, -1):  # different partition.

        clustering = AgglomerativeClustering(
            n_clusters=num, metric="precomputed", linkage="average"
        )  # do clustering.
        clusters = clustering.fit_predict(dsim_mat)

        others["gruppo"] = clusters + cluster_epoch
        cluster_epoch += max_coal + 1

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
                                man,
                                coalition,
                                voting_df,
                                voting_schema,
                                voting_schema_f,
                            )  # compute the new happiness

                            coalition = coalition.iloc[:, :-1]

                            if analyze_core(coal_new_h, "H", "New_H") == True:
                                print(intersection)
                                print(
                                    f"Pushing {alt} made everyone in the group {coal_id} happier, here the new winner:  ",
                                    new_result,
                                )
                                print(f"is it stable?")
                                stable, subcoals = stability_of_coalitions(
                                    coal_new_h,
                                    voting_df,
                                    results,
                                    voting_schema,
                                    voting_schema_f,
                                )
                                print(stable)

                                if stable == True:

                                    coal.append((True, coal_new_h, new_result))
                                else:  # if a coalition is not stable you check if its subcoalitions are (recursive function)

                                    ind = 0
                                    while len(subcoals) > 0:

                                        if len(subcoals[ind].index) > 2:
                                            sb = subcoals[ind].drop(columns="New_H")

                                            sb = sb.rename(
                                                columns={"New_H_subcoal": "New_H"}
                                            )
                                            stable2, sb2 = stability_of_coalitions(
                                                sb,
                                                voting_df,
                                                results,
                                                voting_schema,
                                                voting_schema_f,
                                            )
                                            if stable2 == True:

                                                coal.append(sb)
                                                coal.append(
                                                    (True, subcoals[ind], new_result)
                                                )
                                                subcoals.pop(0)

                                            else:
                                                coal.append(
                                                    (
                                                        False,
                                                        coal_new_h,
                                                        new_result,
                                                        subcoals,
                                                    )
                                                )
                                                subcoals.pop(0)
                                                for ob in sb2:
                                                    subcoals.append(ob)

                                        else:

                                            coal.append(
                                                (True, subcoals[ind], new_result)
                                            )
                                            subcoals.pop(0)

    return build_coalition_table(coal, happiness_level.total)


def build_coalition_table(coalitions, previous_H):
    columns = [
        "coalition_group",
        "voter",
        "strategic_voting",
        "is_stable",
        "new_result",
        "strategic_H",
        "previous_H",
        "strategic_overall_H",
        "previous_overall_H",
    ]

    coalitions_list = []
    for c in coalitions:
        c[1].rename(
            {
                "H": "previous_H",
                "New_H": "strategic_H",
                "manipulation": "strategic_voting",
                "gruppo": "coalition_group",
            },
            axis=1,
            inplace=True,
        )
        c[1]["voter"] = c[1].index
        c[1]["new_result"] = c[2].winner
        c[1]["is_stable"] = c[0]
        coalitions_list.append(c[1])

    if coalitions_list:
        coalitions_df = pd.concat(coalitions_list)
        coalitions_df["previous_overall_H"] = previous_H
        coalitions_df = coalitions_df[columns]
        coalitions_df.reset_index(drop=True, inplace=True)
        return coalitions_df

    return pd.DataFrame(columns=columns)
