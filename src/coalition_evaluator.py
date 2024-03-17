import numpy as np
from itertools import combinations, product, permutations
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
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Analyzing possible coalitions
    for r in range(2, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            for candidate in set(voting_data[0, :]):  # Potential strategic votes
                new_voting_array = voting_data.copy()  # Make a new array for changes
                actual_coalition = []  # Track actual members who changed their vote

                for member in coalition:
                    if candidate in new_voting_array[:, member]:
                        idx = np.where(new_voting_array[:, member] == candidate)[0][0]
                        if idx != 0:  # Only consider if candidate is not already the top choice
                            new_voting_array[:, member] = np.roll(new_voting_array[:, member], -idx)
                            actual_coalition.append(member)

                actual_coalition.sort()  # Ensure the actual coalition list is ordered for consistency
                scenario_key = (tuple(actual_coalition), candidate)  # Define scenario based on actual changes

                if scenario_key not in evaluated_scenarios:
                    new_winner = plurality_outcome(new_voting_array).winner
                    if new_winner != original_winner:
                        new_happiness = HappinessLevel(voting_data, new_winner)
                        happiness_diff = {member: new_happiness.happiness_level_dict[f"voter_{member}"] - original_happiness.happiness_level_dict[f"voter_{member}"] for member in actual_coalition}
                        if all(change > 0 for change in happiness_diff.values()):  # Check all members are happier
                            evaluated_scenarios.add(scenario_key)  # Mark this scenario as evaluated
                            successful_coalitions.append((actual_coalition, candidate, new_winner, happiness_diff))

    # Return after all coalitions are evaluated to avoid repetitive print statements
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
                            successful_coalitions.append((coalition_signature, original_winner, new_winner, happiness_changes))
                            print("New voting configuration:\n", new_voting_array)
                            print(f"Successful Coalition: {coalition}, Strategic Votes: {strategic_votes}, New Winner: {new_winner}")
                            print("Changes in Happiness Levels for Coalition Members:")
                            for voter, change in happiness_changes.items():
                                print(f"Voter {voter}: Happiness Change = {change:.3f}")
                            print("\n")

    return successful_coalitions

def evaluate_coalition_strategic_voting_veto(voting_data: VotingArray):
    num_voters = voting_data.shape[1]
    candidates = list(set(voting_data.flatten()))
    successful_coalitions = []
    unique_coalitions = set()  # To track unique coalitions and prevent duplication
    original_winner = veto_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Analyze potential coalitions of all sizes
    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            # Evaluate changes for each coalition member
            for new_least_preferred_combination in product(candidates, repeat=len(coalition)):
                new_voting_array = voting_data.copy()
                happiness_changes = {}
                strategic_changes = {}
                effective_coalition = []

                for voter_idx, voter in enumerate(coalition):
                    original_least_preferred = voting_data[:, voter][-1]
                    new_least = new_least_preferred_combination[voter_idx]

                    # Apply change only if different from original
                    if original_least_preferred != new_least:
                        updated_preferences = [c for c in voting_data[:, voter] if c != new_least] + [new_least]
                        new_voting_array[:, voter] = updated_preferences
                        strategic_changes[voter] = new_least

                # Determine new outcome
                new_winner = veto_outcome(new_voting_array).winner
                if new_winner != original_winner:
                    # Check happiness impact on each coalition member
                    all_happier = True
                    for voter in coalition:
                        if voter in strategic_changes:  # Only consider voters who made changes
                            temp_happiness = HappinessLevel(voting_data, new_winner)
                            happiness_change = temp_happiness.happiness_level_dict[f'voter_{voter}'] - original_happiness.happiness_level_dict[f'voter_{voter}']
                            happiness_changes[voter] = happiness_change
                            if happiness_change > 0:
                                effective_coalition.append(voter)
                            else:
                                all_happier = False
                                break

                    # Only register the coalition if all involved are happier and necessary
                    if all_happier and effective_coalition:
                        # Create a unique signature for this coalition outcome
                        coalition_signature = (tuple(effective_coalition), new_winner)
                        if coalition_signature not in unique_coalitions:
                            unique_coalitions.add(coalition_signature)
                            successful_coalitions.append((coalition_signature, {v: strategic_changes[v] for v in effective_coalition}, original_winner, new_winner, happiness_changes))
                            print(f"New voting configuration:\n{new_voting_array}")
                            print(f"Effective Coalition: {effective_coalition}, Least Preferred Changes: {strategic_changes}, Original Winner: {original_winner}, New Winner: {new_winner}")
                            print("Changes in Happiness Levels for Coalition Members:")
                            for voter, change in happiness_changes.items():
                                if voter in effective_coalition:
                                    print(f"Voter {voter}: Happiness Change = {change:.3f}")
                            print("\n")

    return successful_coalitions

def evaluate_coalition_strategic_voting_borda(voting_data: VotingArray):
    num_voters = voting_data.shape[1]
    candidates = list(set(voting_data.flatten()))
    successful_coalitions = []
    unique_coalitions = set()  # To track unique coalitions and prevent duplication
    original_winner = borda_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Analyze potential coalitions of all sizes
    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            # Evaluate all possible rearrangements of preferences for each coalition member
            for new_preference_combination in product(permutations(candidates), repeat=len(coalition)):
                new_voting_array = voting_data.copy()
                happiness_changes = {}
                strategic_changes = {}
                effective_coalition = []

                for voter_idx, voter in enumerate(coalition):
                    new_preferences = new_preference_combination[voter_idx]
                    
                    # Check if the new preferences are different from the original
                    if not np.array_equal(voting_data[:, voter], new_preferences):
                        new_voting_array[:, voter] = new_preferences
                        strategic_changes[voter] = new_preferences

                # Determine new outcome
                new_winner = borda_outcome(new_voting_array).winner
                if new_winner != original_winner:
                    # Check happiness impact on each coalition member
                    all_happier = True
                    for voter in coalition:
                        if voter in strategic_changes:  # Only consider voters who made changes
                            temp_happiness = HappinessLevel(voting_data, new_winner)
                            happiness_change = temp_happiness.happiness_level_dict[f'voter_{voter}'] - original_happiness.happiness_level_dict[f'voter_{voter}']
                            happiness_changes[voter] = happiness_change
                            if happiness_change > 0:
                                effective_coalition.append(voter)
                            else:
                                all_happier = False
                                break

                    # Only register the coalition if all involved are happier and necessary
                    if all_happier and effective_coalition:
                        # Create a unique signature for this coalition outcome
                        coalition_signature = (tuple(effective_coalition), new_winner)
                        if coalition_signature not in unique_coalitions:
                            unique_coalitions.add(coalition_signature)
                            successful_coalitions.append((coalition_signature, {v: strategic_changes[v] for v in effective_coalition}, original_winner, happiness_changes))
                            print(f"New voting configuration:\n{new_voting_array}")
                            print(f"Effective Coalition: {effective_coalition}, Strategic Changes: {strategic_changes}, Original Winner: {original_winner}, New Winner: {new_winner}")
                            print("Changes in Happiness Levels for Coalition Members:")
                            for voter, change in happiness_changes.items():
                                if voter in effective_coalition:
                                    print(f"Voter {voter}: Happiness Change = {change:.3f}")
                            print("\n")

    return successful_coalitions




def print_results_coalition_strategic_voting_plurality(successful_coalitions_plurality):
    print("\nSummary of Successful Coalitions under Plurality Voting:")
    # Iterate through each successful coalition
    for (coalition, strategic_vote, new_winner, happiness_diff) in successful_coalitions_plurality:
        # Formatting coalition for printing
        formatted_coalition = ', '.join([f'Voter_{v}' for v in coalition])
        # Print out the results for the coalition
        print(f"\nSuccessful Coalition: {formatted_coalition}, Strategic Vote: {strategic_vote}, New Winner: {new_winner}")
        # Iterate through each member's happiness change within the coalition
        for voter, change in happiness_diff.items():
            print(f"Voter {voter}: Happiness Change = {change:.3f}")
        # Print a newline for better readability between different coalitions
        print()




def print_results_coalition_strategic_voting_for_two(successful_coalitions_for_two):
        # Display the results
    print("\nSummary of Successful Coalitions and their Strategic Votes for VOTINGFORTWO scheme:")
    for (coalition, strategic_votes), original_winner, new_winner, happiness_changes in successful_coalitions_for_two:
        formatted_strategic_votes = ', '.join([f'Voter_{v} = {votes}' for v, votes in strategic_votes])
        print(f"Coalition: {coalition}, Strategic Votes: {formatted_strategic_votes}, Original Winner: {original_winner}, New Winner: {new_winner}, Happiness Changes: {happiness_changes}")




def print_results_coalition_strategic_voting_veto(successful_coalitions_veto):

    print("\nSummary of Successful Coalitions and their Strategic Votes for VETO scheme:")
    for (coalition_signature, strategic_changes_dict, original_winner, new_winner, happiness_changes) in successful_coalitions_veto:
        effective_coalition, new_winner = coalition_signature
        formatted_members = ', '.join([f'Voter_{v}' for v in effective_coalition])
        formatted_changes = ', '.join([f'Voter_{v} = {lp}' for v, lp in strategic_changes_dict.items()])
        formatted_happiness_changes = ', '.join([f'Voter_{v}: {change:.3f}' for v, change in happiness_changes.items()])

        print(f"\nSuccessful Coalition: {formatted_members}")
        print(f"Least Preferred Strategic Change: {formatted_changes}")
        print(f"Original Winner: {original_winner}, New Winner: {new_winner}")
        print(f"Changes in Happiness Levels for Coalition Members: {formatted_happiness_changes}\n")


def print_results_coalition_strategic_voting_borda(successful_coalitions_borda):
    print("\nSummary of Successful Coalitions and their Strategic Votes BORDA:")
    for (coalition_signature, strategic_changes_dict, original_winner, happiness_changes) in successful_coalitions_borda:
        effective_coalition, new_winner = coalition_signature
        formatted_members = ', '.join([f'Voter_{v}' for v in effective_coalition])
        formatted_changes = '; '.join([f'Voter_{v} changes to {" > ".join(map(str, prefs))}' for v, prefs in strategic_changes_dict.items()])
        formatted_happiness_changes = '; '.join([f'Voter_{v}: {change:.3f}' for v, change in happiness_changes.items()])

        print(f"\nSuccessful Coalition Members: {formatted_members}")
        print(f"Strategic Changes (Borda Rankings): {formatted_changes}")
        print(f"Original Winner: {original_winner}, New Winner: {new_winner}")
        print(f"Changes in Happiness Levels for Coalition Members: {formatted_happiness_changes}\n")




















