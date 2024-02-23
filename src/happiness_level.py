import numpy as np

def happiness_level(s: np.array) -> float:
    return .0

def happiness(preferences: np.array, outcome: np.array) -> float:
    """Calculate the happiness level of a voter based on their preferences and the election outcome.

    Args:
        preferences (np.array): An array representing the voter's preferences for the candidates.
        outcome (np.array): An array representing the outcome of the election.

    Returns:
        float: The happiness level of the voter. 0 corresponds to maximum happiness.

    Raises:
        ValueError: If the lengths of the 'preferences' and 'outcome' arrays are not equal.
                    If the 'preferences' array contains duplicate values.
                    If the 'outcome' array contains duplicate values.
                    If the arrays does not contain the same
    """
    
    # Check if the lengths of the 'preferences' and 'outcome' arrays are equal
    if preferences.shape[0] != outcome.shape[0]:
        raise ValueError("The lengths of the 'preferences' and 'outcome' arrays must be equal.")

    # Check if the 'preferences' array contains duplicate values
    if len(np.unique(preferences)) != len(preferences):
        raise ValueError("The 'preferences' array contains duplicate values.")

    # Check if the 'outcome' array contains duplicate values
    if len(np.unique(outcome)) != len(outcome):
        raise ValueError("The 'outcome' array contains duplicate values.")
    
    # Check if both arrays contain the same elements
    if sorted(preferences) != sorted(outcome):
        raise ValueError(" 'preferences' and 'outcome'must contain the same elements.")

    # If all checks pass, call the 'happiness_level_calculation' function to compute the happiness level
    return happiness_level_calculation(preferences=preferences, outcome=outcome)


def happiness_level_calculation(preferences: np.array, outcome: np.array) -> float:
    """_summary_

    @Args:
        preferences (np.array): preference expressed by the voter
        outcome (np.array): outcome of the election

    @Returns:
        float: it returns the happiness of the voter, 0 corresponds to maximum happiness
    """
    
    #initial happiness: note that the highest this number is, the more unhappy this voter is
    happiness = 0
    
    #iteration on the preference list                                        
    for preference_index, preference in enumerate(preferences):    
        #iteration on the outcome list      
        for outcome_index, result in enumerate(outcome):
            if preference == result:
                # i is the index of the preference expressed by the voter while j is the outcome of the election
                #the happiness of the voter is increased by the absolute value of the difference between the index of the vote 
                #expressed and the one in the outcome
                happiness += np.abs(preference_index - outcome_index)
    return happiness