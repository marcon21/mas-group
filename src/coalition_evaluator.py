import numpy as np
from itertools import combinations, product
from src.happiness_level import HappinessLevel
from src.outcomes import plurality_outcome, for_two_outcome, borda_outcome, veto_outcome
from src.utils import VotingArray # or any other needed imports
from itertools import combinations, product

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


def evaluate_coalition_strategic_voting_for_two(voting_data: VotingArray) -> list:
    num_voters = voting_data.shape[1]
    candidates = list(set(voting_data.flatten()))  # Unique candidates from the voting data
    successful_coalitions = []
    original_winner = for_two_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Iterate over potential coalitions and strategic vote combinations
    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            # Each voter might choose a different top two based on their preferences
            all_possible_combinations = product(*[combinations(candidates, 2) for _ in coalition])
            for new_tops in all_possible_combinations:  # New tops now specific to each voter
                new_voting_array = voting_data.copy()
                happiness_changes = {}
                strategic_votes = {}

                for voter_idx, voter in enumerate(coalition):
                    new_top = new_tops[voter_idx]  # Strategic vote specific to this voter
                    original_preferences = list(voting_data[:, voter])
                    new_preferences = [cand for cand in new_top if cand in original_preferences] + [cand for cand in original_preferences if cand not in new_top]
                    new_voting_array[:, voter] = new_preferences
                    strategic_votes[f'Voter_{voter}'] = new_top
                
                # Evaluate if the strategic voting changes the winner
                new_winner = for_two_outcome(new_voting_array).winner
                if new_winner != original_winner:
                    temp_happiness = HappinessLevel(voting_data, new_winner)
                    all_happier = True

                    # Check happiness level changes for all coalition members
                    for voter in coalition:
                        happiness_change = temp_happiness.happiness_level_dict[f"voter_{voter}"] - original_happiness.happiness_level_dict[f"voter_{voter}"]
                        happiness_changes[voter] = happiness_change
                        if happiness_change <= 0:
                            all_happier = False
                            break
                    
                    if all_happier:
                        coalition_signature = (tuple(sorted(coalition)), tuple(strategic_votes.items()))  # Now includes specific votes per voter
                        if coalition_signature not in successful_coalitions:
                            successful_coalitions.append((coalition_signature, new_winner, happiness_changes))
                            print("New voting configuration:\n", new_voting_array)
                            print(f"Successful Coalition: {coalition}, Strategic Votes: {strategic_votes}, New Winner: {new_winner}")
                            print("Changes in Happiness Levels for Coalition Members:")
                            for voter, change in happiness_changes.items():
                                print(f"Voter {voter}: Happiness Change = {change:.3f}")
                            print("\n")

    return successful_coalitions

def evaluate_coalition_strategic_voting_veto(voting_data: VotingArray) -> list:
    num_voters = voting_data.shape[1]
    candidates = list(set(voting_data.flatten()))  # Unique candidates from the voting data
    successful_coalitions = []
    original_winner = veto_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Iterate over potential coalitions and strategic least preferred candidate changes
    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            for new_least_preferred in product(candidates, repeat=len(coalition)):  # New least preferred candidate for each coalition member
                new_voting_array = voting_data.copy()
                happiness_changes = {}
                strategic_votes = {}

                for voter_idx, voter in enumerate(coalition):
                    new_least = new_least_preferred[voter_idx]  # New least preferred candidate for this voter
                    if voting_data[-1, voter] != new_least:  # Check if there's a change in the least preferred candidate
                        # Change the least preferred candidate
                        new_preferences = [cand for cand in voting_data[:, voter] if cand != new_least] + [new_least]
                        new_voting_array[:, voter] = new_preferences
                        strategic_votes[f'Voter_{voter}'] = new_least
                
                # Evaluate if the strategic voting changes the winner
                new_winner = veto_outcome(new_voting_array).winner
                if new_winner != original_winner:
                    temp_happiness = HappinessLevel(voting_data, new_winner)
                    all_happier = True

                    # Check happiness level changes for all coalition members
                    for voter in coalition:
                        happiness_change = temp_happiness.happiness_level_dict[f"voter_{voter}"] - original_happiness.happiness_level_dict[f"voter_{voter}"]
                        happiness_changes[voter] = happiness_change
                        if happiness_change <= 0:
                            all_happier = False
                            break
                    
                    if all_happier:
                        coalition_signature = (tuple(sorted(coalition)), tuple(strategic_votes.items()))
                        if coalition_signature not in successful_coalitions:
                            successful_coalitions.append((coalition_signature, new_winner, happiness_changes))
                            print("New voting configuration:\n", new_voting_array)
                            print(f"Successful Coalition: {coalition}, Least Preferred Changes: {strategic_votes}, New Winner: {new_winner}")
                            print("Changes in Happiness Levels for Coalition Members:")
                            for voter, change in happiness_changes.items():
                                print(f"Voter {voter}: Happiness Change = {change:.3f}")
                            print("\n")

    return successful_coalitions



















