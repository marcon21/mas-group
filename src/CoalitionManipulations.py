import numpy as np
import pandas as pd
import hashlib
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from src import utils
from src.outcomes import plurality_outcome
from src.outcomes import borda_outcome
from src.outcomes import for_two_outcome
from src.outcomes import veto_outcome
from src.happiness_level import HappinessLevel


def get_df_hash(df):
    df_bytes = df.to_json().encode()
    return hashlib.md5(df_bytes).hexdigest()


def remove_elements_above_or_equal_index(lst, index):
    if index < 0 or index >= len(lst):
        return lst

    return lst[:index]


def burying(new_poss_coal, results):
    mans = []

    for index, row in new_poss_coal.iterrows():
        pref = list(row)
        ind = pref.index(results.winner)
        pref.pop(ind)
        pref.append(results.winner)
        mans.append(pref)
    return [mans]


def compromise(new_poss_coal, results):
    better_op = []
    mans = []

    for index, row in new_poss_coal.iterrows():
        pref = list(row)
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
                    pref = list(row)
                    ind = pref.index(alt)
                    pref.pop(ind)
                    pref.insert(0, alt)
                    man.append(pref)

                mans.append(man)
    else:
        for alt in better_op[0]:

            man = []  # initialize the list with the voters manipulations

            for index, row in new_poss_coal.iterrows():
                pref = list(row)
                ind = pref.index(alt)
                pref.pop(ind)
                pref.insert(0, alt)
                man.append(pref)

            mans.append(man)

    return mans


def find_new_happiness(
    manipulations, coalition, voting_df, voting
):  # Function that for a manipulation of a coalition gives you the new happiness values
    n_cand = len(voting_df.columns) - 1
    coalition["manipulation"] = manipulations
    indici = coalition.index
    df = pd.DataFrame(
        manipulations, index=indici, columns=[i for i in range(0, n_cand)]
    )

    new_voting_df = voting_df.copy()
    new_voting_df.loc[indici] = df
    if voting == "plurality":
        new_results = plurality_outcome(new_voting_df.iloc[:, :n_cand].values.T)
    elif voting == "borda":

        new_results = borda_outcome(new_voting_df.iloc[:, :n_cand].values.T)
    elif voting == "voting_for_two":

        new_results = for_two_outcome(new_voting_df.iloc[:, :n_cand].values.T)
    elif voting == "veto":

        new_results = veto_outcome(new_voting_df.iloc[:, :n_cand].values.T)

    diz = HappinessLevel(
        voting_df.iloc[:, :n_cand].values.T, new_results.winner
    ).happiness_level_dict
    New_Happiness_Levels = pd.DataFrame.from_dict(
        diz, orient="index", columns=["New_H"]
    )  # compute the happiness of the new results with respect of the real preferences
    coalition = pd.merge(
        coalition, New_Happiness_Levels, left_index=True, right_index=True
    )

    return coalition, new_results


def find_new_happiness2(
    manipulations, coalition, voting_df, voting
):  # Function that for a manipulation of a coalition gives you the new happiness values
    n_cand = len(voting_df.columns) - 1
    coalition["manipulation"] = manipulations
    indici = coalition.index
    df = pd.DataFrame(
        manipulations, index=indici, columns=[i for i in range(0, n_cand)]
    )
    new_voting_df = voting_df.copy()

    new_voting_df.loc[indici] = df
    if voting == "plurality":
        new_results = plurality_outcome(new_voting_df.iloc[:, :n_cand].values.T)
    elif voting == "borda":
        new_results = borda_outcome(new_voting_df.iloc[:, :n_cand].values.T)
    elif voting == "voting_for_two":
        new_results = for_two_outcome(new_voting_df.iloc[:, :n_cand].values.T)
    elif voting == "veto":
        new_results = for_two_outcome(new_voting_df.iloc[:, :n_cand].values.T)

    diz = HappinessLevel(
        voting_df.iloc[:, :n_cand].values.T, new_results.winner
    ).happiness_level_dict
    New_Happiness_Levels = pd.DataFrame.from_dict(
        diz, orient="index", columns=["New_H_subcoal"]
    )  # compute the happiness of the new results with respect of the real preferences
    coalition = pd.merge(
        coalition, New_Happiness_Levels, left_index=True, right_index=True
    )

    return coalition, new_results


def analyze_core(coalition, var1, var2):  # Analize if inside or not the code

    real_happ = coalition[var1]
    fake_happ = coalition[var2]

    comparison_result = [fake > real for real, fake in zip(real_happ, fake_happ)]

    if all(comparison_result):
        return True
    else:
        return False


def stability_of_coalitions(coal, voting_df, results, voting, strategy):

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
            if strategy == "compromising":

                mans = compromise(new_poss_coal.iloc[:, 0:-4], results)
            else:
                mans = burying(new_poss_coal.iloc[:, 0:-4], results)

            for man in mans:
                coal_new_h, new_result = find_new_happiness2(
                    man, new_poss_coal.iloc[:, :-1], voting_df, voting
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
    voting_df, happiness_level, results, voting="plurality", strategy="compromising"
):

    win = results.winner
    voting_df["H"] = happiness_level
    others = voting_df[voting_df[0] != win]
    # Creating Dissimilarity Matrix
    rankings = np.array(others.T)
    losers = 0
    for valore in happiness_level:
        if valore != 1:
            losers += 1
    if len(others.index) > 1:

        if len(others.index) == 2:
            cor_mat, _ = spearmanr(rankings[:-1])
            cor_mat2 = np.zeros((2, 2))
            cor_mat2[0, 1] = cor_mat
            cor_mat2[1, 0] = cor_mat
            cor_mat2[1, 1] = 1
            cor_mat2[0, 0] = 1
            dsim_mat = np.around(1 - cor_mat2, decimals=4)
        else:

            cor_mat, _ = spearmanr(rankings[:-1])
            dsim_mat = np.around(1 - cor_mat, decimals=4)
            np.fill_diagonal(dsim_mat, 0)

    # find stable coalitions
    coal = []
    coal_index = {}
    New_Results = []
    for num in range(losers, 0, -1):  # different partition.
        if num > 1:
            clustering = AgglomerativeClustering(
                n_clusters=num, metric="precomputed", linkage="average"
            )  # do clustering.
            clusters = clustering.fit_predict(dsim_mat)

        else:
            clusters = np.zeros(len(others.index))

        others["gruppo"] = clusters
        for coal_id, coalition in others.groupby("gruppo"):

            if get_df_hash(coalition.iloc[:, :-2]) not in coal_index:

                coal_index[get_df_hash(coalition.iloc[:, :-2])] = coalition
                if strategy == "compromising":
                    mans = compromise(coalition.iloc[:, :-2], results)
                else:
                    mans = burying(coalition.iloc[:, 0:-2], results)

                for man in mans:
                    coal_new_h, new_result = find_new_happiness(
                        man, coalition, voting_df, voting
                    )  # compute the new happiness

                    coalition = coalition.iloc[:, :-1]

                    if analyze_core(coal_new_h, "H", "New_H") == True:

                        stable, subcoals = stability_of_coalitions(
                            coal_new_h, voting_df, results, voting, strategy
                        )

                        if stable == True:
                            coal.append((True, coal_new_h, new_result))

                        else:  # if a coalition is not stable you check if its subcoalitions are (recursive function)

                            ind = 0
                            while len(subcoals) > 0:

                                if len(subcoals[ind].index) > 2:
                                    sb = subcoals[ind].drop(columns="New_H")

                                    sb = sb.rename(columns={"New_H_subcoal": "New_H"})
                                    stable2, sb2 = stability_of_coalitions(
                                        sb, voting_df, results, voting, strategy
                                    )
                                    if stable2 == True:
                                        coal.append(
                                            (
                                                True,
                                                subcoals[ind],
                                                new_result,
                                            )
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
                                    coal.append((True, subcoals[ind], new_result))
                                    subcoals.pop(0)

    return coal


def coalition_dataframe(coals, voting_array, valori_original_happiness):
    df = pd.DataFrame(
        columns=[
            "Number_voters",
            "voters",
            # "number_manipulations",
            "strategic_voting_risk",
            "overall_happiness_change_inside_coalition",
            "overall_happiness_change_system",
        ]
    )

    i = 0

    # dix = {}
    diz_coal = {}
    for el in coals:
        dataset = el[1]
        indici = tuple(dataset.index)
        # if indici not in list(df["voters"]):
        diz_coal[get_df_hash(dataset)] = dataset
        dataset["diff"] = dataset["New_H"] - dataset["H"]
        df.loc[i, "Number_voters"] = len(indici)
        df.loc[i, "voters"] = tuple(indici)
        # df.loc[i, "number_manipulations"] = 1
        df.loc[i, "strategic_voting_risk"] = dataset["diff"].max()
        df.loc[i, "overall_happiness_change_inside_coalition"] = (
            dataset["diff"].sum() / df.loc[i, "Number_voters"]
        )
        # dix[tuple(indici)] = i

        new_happiness_level = HappinessLevel(voting_array, el[2].winner)
        valori_new_happiness = np.array(
            list(new_happiness_level.happiness_level_dict.values())
        )

        overall_diff = np.sum(valori_new_happiness - valori_original_happiness) / len(
            valori_new_happiness
        )
        df.loc[i, "overall_happiness_change_system"] = overall_diff
        i += 1
        """
        else:
            if get_df_hash(dataset) not in diz_coal:

                diz_coal[get_df_hash(dataset)] = dataset
                dataset["diff"] = dataset["New_H"] - dataset["H"]
                df.loc[dix[tuple(indici)], "number_manipulations"] += 1
                # if there are more than 1 possible manipulations compute the mean
                df.loc[dix[tuple(indici)], "strategic_voting_risk"] += dataset[
                    "diff"
                ].max()
                df.loc[dix[tuple(indici)], "strategic_voting_risk"] = (
                    df.loc[dix[tuple(indici)], "strategic_voting_risk"]
                    / df.loc[dix[tuple(indici)], "number_manipulations"]
                )
                df.loc[
                    dix[tuple(indici)], "overall_happiness_change_inside_coalition"
                ] += dataset["diff"].sum() / len(indici)
                df.loc[
                    dix[tuple(indici)], "overall_happiness_change_inside_coalition"
                ] = (
                    df.loc[
                        dix[tuple(indici)], "overall_happiness_change_inside_coalition"
                    ]
                    / df.loc[dix[tuple(indici)], "number_manipulations"]
                )

                new_happiness_level = HappinessLevel(voting_array, el[2].winner)
                valori_new_happiness = np.array(
                    list(new_happiness_level.happiness_level_dict.values())
                )

                overall_diff = np.sum(
                    valori_new_happiness - valori_original_happiness
                ) / len(valori_new_happiness)
                df.loc[
                    dix[tuple(indici)], "overall_happiness_change_system"
                ] += overall_diff
                df.loc[dix[tuple(indici)], "overall_happiness_change_system"] = (
                    df.loc[dix[tuple(indici)], "overall_happiness_change_system"]
                    / df.loc[dix[tuple(indici)], "number_manipulations"]
                )
            """
    return df
