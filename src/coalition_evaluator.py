import numpy as np
from itertools import chain, combinations, product, permutations
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
    candidates = list(set(voting_data.flatten()))
    successful_coalitions = []
    original_winner = for_two_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    def is_new_coalition_valid(new_coalition, successful_coalitions):
        new_coalition_set = set(new_coalition)
        for existing_coalition, _, _, _, _ in successful_coalitions:
            if set(existing_coalition).issubset(new_coalition_set):
                # The new coalition contains an entire existing successful coalition
                return False
        return True

    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            all_possible_combinations = product(*[combinations(candidates, 2) for _ in coalition])
            for new_tops in all_possible_combinations:
                new_voting_array = voting_data.copy()
                happiness_changes = {}
                strategic_votes = {}

                for voter_idx, voter in enumerate(coalition):
                    new_top = new_tops[voter_idx]
                    new_preferences = [cand for cand in new_top if cand in candidates] + [cand for cand in candidates if cand not in new_top]
                    new_voting_array[:, voter] = new_preferences
                    strategic_votes[voter] = new_top
                
                new_winner = for_two_outcome(new_voting_array).winner
                if new_winner != original_winner:
                    temp_happiness = HappinessLevel(voting_data, new_winner)
                    all_happier = True

                    for voter in coalition:
                        happiness_change = temp_happiness.happiness_level_dict[f"voter_{voter}"] - original_happiness.happiness_level_dict[f"voter_{voter}"]
                        happiness_changes[voter] = happiness_change
                        if happiness_change <= 0:
                            all_happier = False
                            break

                    if all_happier and is_new_coalition_valid(coalition, successful_coalitions):
                        successful_coalitions.append((coalition, strategic_votes, original_winner, new_winner, happiness_changes))

    return successful_coalitions





def evaluate_coalition_strategic_voting_veto(voting_data: VotingArray):
    num_voters = voting_data.shape[1]
    candidates = list(set(voting_data.flatten()))
    successful_coalitions = []
    unique_coalitions = set()  # To track unique coalitions and prevent duplication
    original_winner = veto_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Generate all possible subsets for coalition validation
    def all_subsets(ss):
        return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))

    # Analyze potential coalitions of all sizes
    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            for new_least_preferred_combination in product(candidates, repeat=len(coalition)):
                new_voting_array = voting_data.copy()
                happiness_changes = {}
                strategic_changes = {}
                effective_coalition = []

                # Apply strategic changes and check for new outcomes
                for voter_idx, voter in enumerate(coalition):
                    original_least_preferred = voting_data[:, voter][-1]
                    new_least = new_least_preferred_combination[voter_idx]
                    if original_least_preferred != new_least:
                        updated_preferences = [c for c in voting_data[:, voter] if c != new_least] + [new_least]
                        new_voting_array[:, voter] = updated_preferences
                        strategic_changes[voter] = new_least

                new_winner = veto_outcome(new_voting_array).winner
                if new_winner != original_winner:
                    temp_happiness = HappinessLevel(voting_data, new_winner)
                    all_happier = True

                    # Check if coalition members are happier with the change
                    for voter in coalition:
                        if voter in strategic_changes:
                            happiness_change = temp_happiness.happiness_level_dict[f'voter_{voter}'] - original_happiness.happiness_level_dict[f'voter_{voter}']
                            happiness_changes[voter] = happiness_change
                            if happiness_change <= 0:
                                all_happier = False
                                break
                            effective_coalition.append(voter)

                    # Verify if a smaller subset already succeeded
                    subset_valid = True
                    for subset in all_subsets(effective_coalition):
                        subset_signature = (tuple(sorted(subset)), new_winner)
                        if subset_signature in unique_coalitions:
                            subset_valid = False
                            break

                    # Register new successful coalition
                    if all_happier and subset_valid:
                        coalition_signature = (tuple(sorted(effective_coalition)), new_winner)
                        unique_coalitions.add(coalition_signature)
                        successful_coalitions.append((coalition_signature, {v: strategic_changes[v] for v in effective_coalition}, original_winner, new_winner, happiness_changes))

    return successful_coalitions


def evaluate_coalition_strategic_voting_borda(voting_data: VotingArray):
    num_voters = voting_data.shape[1]
    candidates = list(set(voting_data.flatten()))
    successful_coalitions = []
    unique_coalitions = set()
    original_winner = borda_outcome(voting_data).winner
    original_happiness = HappinessLevel(voting_data, original_winner)

    # Generate all possible subsets for coalition validation
    # def all_subsets(ss):
    #     return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))

    # Iterate over potential coalitions
    for r in range(1, num_voters + 1):
        for coalition in combinations(range(num_voters), r):
            # Check all possible rearrangements for each member of the coalition
            for new_prefs_combination in product(permutations(candidates), repeat=len(coalition)):
                new_voting_array = voting_data.copy()
                happiness_changes = {}
                strategic_changes = {}
                effective_coalition = []

                # Apply the new preferences if they represent a change
                for voter_idx, voter in enumerate(coalition):
                    new_preferences = list(new_prefs_combination[voter_idx])
                    if list(voting_data[:, voter]) != new_preferences:
                        new_voting_array[:, voter] = new_preferences
                        strategic_changes[voter] = new_preferences

                # Check if changes influence the outcome
                new_winner = borda_outcome(new_voting_array).winner
                if new_winner != original_winner:
                    temp_happiness = HappinessLevel(voting_data, new_winner)
                    all_happier = True

                    for voter in coalition:
                        if voter in strategic_changes:
                            happiness_change = temp_happiness.happiness_level_dict[f'voter_{voter}'] - original_happiness.happiness_level_dict[f'voter_{voter}']
                            if happiness_change <= 0:
                                all_happier = False
                                break
                            else:
                                happiness_changes[voter] = happiness_change
                                effective_coalition.append(voter)
                        
                    # Verify if a smaller subset already succeeded
                    # subset_valid = True
                    # for subset in all_subsets(effective_coalition):
                    #     subset_signature = (tuple(sorted(subset)), new_winner)
                    #     if subset_signature in unique_coalitions:
                    #         subset_valid = False
                    #         break

                    # Register successful coalitions ensuring no redundant members and no duplicate coalitions
                    if all_happier and effective_coalition:
                        coalition_signature = (tuple(sorted(effective_coalition)), new_winner)
                        if coalition_signature not in unique_coalitions:
                            unique_coalitions.add(coalition_signature)
                            successful_coalitions.append((coalition_signature, {v: strategic_changes[v] for v in effective_coalition}, original_winner, new_winner, happiness_changes))

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




def print_results_coalition_strategic_voting_for_two(successful_coalitions):
    print("\nSummary of Successful Coalitions and their Strategic Votes (Voting for Two):")

    for coalition_data in successful_coalitions:
        coalition, strategic_votes, original_winner, new_winner, happiness_changes = coalition_data

        # Format the coalition for printing
        formatted_coalition = ', '.join([f'Voter_{v}' for v in coalition])

        # Format the strategic votes for printing
        formatted_strategic_votes = ', '.join([f'Voter_{v}: {votes}' for v, votes in strategic_votes.items()])

        # Format the happiness changes for printing
        formatted_happiness_changes = ', '.join([f'Voter_{v}: {change:.3f}' for v, change in happiness_changes.items()])

        # Print the coalition results
        print(f"\nSuccessful Coalition: {formatted_coalition}")
        print(f"Strategic Votes Changes: {formatted_strategic_votes}")
        print(f"Original Winner: {original_winner}, New Winner: {new_winner}")
        print(f"Changes in Happiness Levels for Coalition Members: {formatted_happiness_changes}\n")





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
    print("\nSummary of Successful Coalitions and their Strategic Votes (Borda):")
    for (coalition_signature, strategic_changes_dict, original_winner, new_winner, happiness_changes) in successful_coalitions_borda:
        effective_coalition, _ = coalition_signature
        formatted_members = ', '.join([f'Voter_{v}' for v in effective_coalition])
        formatted_changes = ', '.join([f'Voter_{v} changes: {changes}' for v, changes in strategic_changes_dict.items()])
        formatted_happiness_changes = ', '.join([f'Voter_{v}: {change:.3f}' for v, change in happiness_changes.items()])

        # Print the details of each successful coalition
        print(f"Coalition Members: {formatted_members}")
        print(f"Strategic Changes: {formatted_changes}")
        print(f"Original Winner: {original_winner}, New Winner: {new_winner}")
        print(f"Changes in Happiness Levels for Coalition Members: {formatted_happiness_changes}\n")



















