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
