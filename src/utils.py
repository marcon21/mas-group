import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.outcomes import Result


class VotingArray(np.ndarray):
    def to_pandas(self):
        p, v = self.shape
        voting_table = pd.DataFrame(
            self,
            columns=[f"voter_{i}" for i in range(v)],
            index=[f"preference_{i}" for i in range(p)],
        )
        return voting_table


def read_voting(file_path: str, table_name: str = "voting") -> VotingArray:
    with open(file_path) as f:
        voting = json.load(f)[table_name]
        voting = np.array(voting).view(VotingArray)

    return voting


def random_voting(n_voters: int, n_candidates: int) -> VotingArray:
    """
    Generates a random voting table with n_voters and n_candidates, the candidates are the letter of the alphabet
    """

    def generate_candidates(n):
        """
        Generates a list of candidates with n elements
        """
        from string import ascii_uppercase
        import itertools

        # Python Generator to generate candidate names
        def iter_all_strings():
            size = 1
            while True:
                for s in itertools.product(ascii_uppercase, repeat=size):
                    yield "".join(s)
                size += 1

        # Get the first n elements of the iterator
        return list(itertools.islice(iter_all_strings(), n))

    candidates = generate_candidates(n_candidates)
    voting = np.empty(
        (n_candidates, n_voters), dtype=f"U{len(max(candidates, key=len))}"
    )

    # Populate each column with a shuffled version of the candidates list
    for col in range(n_voters):
        np.random.shuffle(candidates)
        voting[:, col] = candidates

    return voting.view(VotingArray)

def generate_scheme(tipology: str, candidates: float) -> np.ndarray:
    """
    Generate a voting scheme based on the given tipology and number of candidates.

    Parameters:
        tipology (str): The type of voting scheme to generate. It can be one of the following: "borda", "plurality", "veto", "for_two".
        candidates (int): The number of candidates in the election.

    Returns:
        np.ndarray: The generated voting scheme as a NumPy array.
    """
    if tipology == "borda":
        return np.arange(candidates - 1, -1, -1)
    elif tipology == "plurality":
        return np.array([1] + [0] * (candidates - 1))
    elif tipology == "veto":
        return np.array([1] * (candidates - 1) + [0])
    elif tipology == "for_two":
        return np.array([1, 1] + [0] * (candidates - 2))
    else:
        raise ValueError("Invalid tipology. It must be one of: 'borda', 'plurality', 'veto', 'for_two'")

def condition_checker(voting_scheme: np.ndarray, voting_solution: pd.DataFrame):
    """
    Check conditions for valid input: type check, length check and two candidates condition. Valid voting_scheme is checked later.

    Parameters:
        voting_scheme (np.ndarray): The NumPy array representing the voting scheme.
        voting_solution (pd.DataFrame): The DataFrame containing the voting solutions.
    """
    # Check if voting_solution is a DataFrame
    if not isinstance(voting_solution, VotingArray):
        raise TypeError("voting_solution must be a Voting Array")
        
    # Check if voting_scheme is a NumPy array
    if not isinstance(voting_scheme, np.ndarray):
        raise TypeError("Voting Scheme must be a Np.ndArray")
        
    # Check if the length of voting_scheme matches the number of candidates
    if voting_scheme.shape[0] != voting_solution.shape[0]:
        raise ValueError("Length of voting scheme must be equal to the number of candidates")
        
    # Check if there are at least two candidates
    if voting_solution.shape[1] < 2:
        raise ValueError("There must be at least two candidates")
    
    
def display_vote_graph(result:Result):
    """
    Display a graph showing the winner and the votes received by each candidate.

    Args:
    - result (Result): A Result object containing the voting results.
    """
    # Extract candidate names and corresponding votes
    candidates = list(result.keys())
    votes = list(result.values())

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(candidates, votes, color='skyblue')

    # Highlight the winner
    winner = result.winner
    winner_index = candidates.index(winner)
    plt.bar(winner_index, votes[winner_index], color='orange')

    # Add labels and title
    plt.xlabel('Candidates')
    plt.ylabel('Votes')
    plt.title('Voting Results')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Votes', 'Winner'])

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    r = random_voting(5, 27)
    print(r.to_pandas())
