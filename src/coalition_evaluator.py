import numpy as np
from itertools import combinations
from src.happiness_level import HappinessLevel
from src.outcomes import plurality_outcome, for_two_outcome, borda_outcome, veto_outcome
from src.utils import VotingArray # or any other needed imports

def evaluate_coalition_strategic_voting_plurality(voting_data: VotingArray) -> list:
    num_voters = voting_data.shape[1]
    successful_coalitions = []
    evaluated_scenarios = set()  # Keep track of evaluated scenarios to avoid duplicates

    # Original setup
    original_winner = plurality_outcome(voting_data).winner
    print(voting_data)
    print("original winner: " + original_winner + "\n")
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Analyzing possible coalitions
    # Analyzing possible coalitions
    for r in range(2, num_voters + 1):
        for initial_coalition in combinations(range(num_voters), r):
            for candidate in set(voting_data[0, :]):  # Potential strategic votes
                # Unique identifier for a coalition scenario based on its members and the strategic candidate
                coalition_key = (tuple(sorted(initial_coalition)), candidate)
                # Check if this scenario was already evaluated
                if coalition_key in evaluated_scenarios:
                    continue  # Skip if already evaluated

                new_voting_array = voting_data.copy()  # Make a new array for changes
                actual_coalition = []  # Track actual members who changed their vote

                for member in initial_coalition:
                    if candidate in new_voting_array[:, member]:
                        idx = np.where(new_voting_array[:, member] == candidate)[0][0]
                        if idx != 0:  # Only consider if candidate is not already the top choice
                            new_voting_array[:, member] = np.roll(new_voting_array[:, member], -idx)
                            actual_coalition.append(member)  # Add to actual coalition

                if actual_coalition:
                    new_winner = plurality_outcome(new_voting_array).winner
                    new_happiness = HappinessLevel(voting_data, new_winner)

                    # Check if all actual coalition members are happier with the new outcome
                    if all(new_happiness.happiness_level_dict[f"voter_{member}"] > original_happiness.happiness_level_dict[f"voter_{member}"] for member in actual_coalition):
                        successful_coalitions.append((actual_coalition, candidate, new_winner))
                        evaluated_scenarios.add(coalition_key)  # Mark this scenario as evaluated
                        print(new_voting_array)
                        print(f"Successful Coalition: {actual_coalition}, Strategic Vote: {candidate}, New Winner: {new_winner}")
                        print("Changes in Happiness Levels for Coalition Members:")
                        for member in actual_coalition:
                            happiness_diff = new_happiness.happiness_level_dict[f"voter_{member}"] - original_happiness.happiness_level_dict[f"voter_{member}"]
                            print(f"Voter {member}: Happiness Change = {happiness_diff:.3f}")
                        print("\n")

    return successful_coalitions


def evaluate_coalition_strategic_voting_for_two(voting_data: VotingArray) -> list:
    num_voters = voting_data.shape[1]
    successful_coalitions = []
    evaluated_scenarios = set()  # Keep track of evaluated scenarios to avoid duplicates

    # Original setup
    original_winner = for_two_outcome(voting_data).winner  # Using plurality outcome for the final winner
    print(voting_data)
    print("original winner: " + original_winner + "\n")
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Analyzing possible coalitions
    for r in range(2, num_voters + 1):
        for initial_coalition in combinations(range(num_voters), r):
            # Consider all possible candidate combinations as new top two choices for the coalition
            for new_top_two in combinations(set(voting_data.flatten()), 2):
                # Check if this scenario has been evaluated
                coalition_key = (tuple(sorted(initial_coalition)), new_top_two)
                if coalition_key in evaluated_scenarios:
                    continue  # Skip if already evaluated
                
                new_voting_array = voting_data.copy()  # Make a new array for changes
                actual_coalition = []  # Track actual members who changed their vote

                for member in initial_coalition:
                    # Check if any of the new top two choices differ from their original top two
                    if not set(new_top_two).issubset(voting_data[:2, member]):
                        # Reassign their top two preferences
                        remaining_choices = list(set(voting_data[:, member]) - set(new_top_two))
                        new_preferences = list(new_top_two) + remaining_choices
                        new_voting_array[:, member] = new_preferences
                        actual_coalition.append(member)

                if actual_coalition:  # Proceed only if there was an actual change
                    # Evaluate outcome based on new potential top votes
                    new_winner = for_two_outcome(new_voting_array).winner
                    new_happiness = HappinessLevel(voting_data, new_winner)

                    # Check if all actual coalition members are happier with the new outcome
                    if all(new_happiness.happiness_level_dict[f"voter_{member}"] > original_happiness.happiness_level_dict[f"voter_{member}"] for member in actual_coalition):
                        successful_coalitions.append((actual_coalition, new_top_two, new_winner))
                        evaluated_scenarios.add(coalition_key)  # Mark this scenario as evaluated
                        print("New voting configuration:\n", new_voting_array)
                        print(f"Successful Coalition: {actual_coalition}, Strategic Votes: {new_top_two}, New Winner: {new_winner}")
                        print("Changes in Happiness Levels for Coalition Members:")
                        for member in actual_coalition:
                            happiness_diff = new_happiness.happiness_level_dict[f"voter_{member}"] - original_happiness.happiness_level_dict[f"voter_{member}"]
                            print(f"Voter {member}: Happiness Change = {happiness_diff:.3f}")
                        print("\n")

    return successful_coalitions