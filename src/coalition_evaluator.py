import numpy as np
from itertools import combinations, product
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
                        print(f"Successful Coalition\n: {actual_coalition}, Strategic Vote: {candidate}, New Winner: {new_winner}")
                        print("Changes in Happiness Levels for Coalition Members:")
                        for member in actual_coalition:
                            happiness_diff = new_happiness.happiness_level_dict[f"voter_{member}"] - original_happiness.happiness_level_dict[f"voter_{member}"]
                            print(f"Voter {member}: Happiness Change = {happiness_diff:.3f}")
                        print("\n")

    return successful_coalitions


from itertools import combinations, product

def evaluate_coalition_strategic_voting_for_two(voting_data: VotingArray) -> list:
    num_voters = voting_data.shape[1]
    successful_coalitions = []
    original_winner = for_two_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Iterate over potential coalitions and their new top two choices
    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            for new_top in combinations(['A', 'B', 'C', 'D'], 2):  # Assuming these are all candidates
                new_voting_array = voting_data.copy()
                coalition_changed = False
                happiness_increased = True

                for voter in coalition:
                    if voting_data[:, voter][0] != original_winner:
                        # Reorder the voter's preferences based on new_top while keeping the original preferences
                        original_preferences = list(voting_data[:, voter])
                        new_preferences = [cand for cand in new_top if cand in original_preferences] + [cand for cand in original_preferences if cand not in new_top]
                        new_voting_array[:, voter] = new_preferences
                        coalition_changed = True

                if coalition_changed:
                    new_winner = for_two_outcome(new_voting_array).winner
                    if new_winner != original_winner:
                        for voter in coalition:
                            temp_happiness = HappinessLevel(voting_data, new_winner).happiness_level_dict[f"voter_{voter}"]
                            if temp_happiness <= original_happiness.happiness_level_dict[f"voter_{voter}"]:
                                happiness_increased = False
                                break

                        if happiness_increased:
                            # Convert the coalition and strategic votes into a unique identifier
                            coalition_signature = (tuple(sorted(coalition)), new_top, new_winner)
                            if coalition_signature not in successful_coalitions:
                                successful_coalitions.append(coalition_signature)
                                print("New voting configuration:\n", new_voting_array)
                                print(f"Successful Coalition: {coalition}, Strategic Votes: {new_top}, New Winner: {new_winner}")
                                print("Changes in Happiness Levels for Coalition Members:")
                                for voter in coalition:
                                    happiness_diff = temp_happiness - original_happiness.happiness_level_dict[f"voter_{voter}"]
                                    print(f"Voter {voter}: Happiness Change = {happiness_diff:.3f}")
                                print("\n")

    return successful_coalitions












