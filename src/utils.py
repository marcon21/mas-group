import json
import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    r = random_voting(5, 27)
    print(r.to_pandas())
