import numpy as np
from src.outcomes import Result
from src.happiness_level import HappinessLevel
from src.utils import display_vote_graph
from src.strategic_voting_risk import StrategicVoting


class Analysis:
    def __init__(self, outcome: Result, happiness: HappinessLevel, total_happiness: float, strategies: StrategicVoting, risk: float):
        """
        Initialize an Analysis object with the provided parameters.

        Parameters:
            outcome (float): The outcome of the voting analysis.
            happiness (pd.Series): A Series containing the happiness level for each voter.
            total_happiness (float): The total happiness calculated based on the individual happiness levels.
            strategies (lStrategicVoting): 
            risk (int): The calculated risk associated with the voting analysis.
        """
        self.outcome = outcome
        self.happiness = happiness
        self.total_happiness = total_happiness
        self.strategies = strategies
        self.risk = risk
        
    def represent(self):
    
        display_vote_graph(self.outcome)
        self.happiness.plot()
        print("Total happiness is:",self.total_happiness)
        print(self.strategies.all)
        print("Total risk is:",self.total_happiness)
        
