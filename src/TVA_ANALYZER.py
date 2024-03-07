import pandas as pd
import numpy as np
from src.outcomes import *
from src.Analysis import Analysis
from src.happiness_level import HappinessLevel
from src.strategic_voting_risk import StrategicVoting
from src.happiness_level import VotingArray
from src.utils import condition_checker
from typing import Callable


class TVA_ANALYZER:
    
    def __init__(self, voting_scheme: np.ndarray, voting_solution: VotingArray):
        """
        Initialize the TVA_ANALYZER object with the provided voting scheme and voting solution.

        Parameters:
            voting_scheme (np.ndarray): NumPy array representing the voting scheme.
                The elements in the array represent the voting weights assigned to each candidate.
            voting_solution (pd.DataFrame): DataFrame containing the voting solutions.
                Each row represents a voter, and each column represents a candidate's preference.
        """
        # Check the conditions for valid input
        condition_checker(voting_scheme, voting_solution)
        
        # Determine the voting outcome based on the provided scheme
        # and assign it to the object's attribute
        if all(voting_scheme[i] == len(voting_scheme) - i for i in range(len(voting_scheme))):
            self.voting_scheme = borda_outcome
        elif voting_scheme[0] == 1 and all(voting_scheme[i] == 0 for i in range(1, len(voting_scheme))):
            self.voting_scheme = plurality_outcome
        elif voting_scheme[0] == 1 and voting_scheme[1] == 1 and all(voting_scheme[i] == 0 for i in range(2, len(voting_scheme))):
            self.voting_scheme = for_two_outcome
        elif all(voting_scheme[i] == 1 for i in range(len(voting_scheme) - 1)) and voting_scheme[-1] == 0:
            self.voting_scheme = veto_outcome
        else:
            raise ValueError("Invalid voting scheme")

        # Assign the voting solution to the object's attribute
        self.voting_solution = voting_solution

    def analyze(self) -> 'Analysis':
        """
        Analyze the voting solution and return the analysis results.

        Returns:
            Analysis: An object containing the outcome, happiness levels, total happiness, strategic voting strategy, and associated risk.
        """
        # Calculate the outcome of the voting system
        outcome = self.calculate_outcome()

        # Calculate the happiness level of each voter based on the outcome
        happiness = self.calculate_happiness(outcome)

        # Calculate the total happiness based on individual happiness levels
        total_happiness = self.calculate_total_happiness(happiness)

        # Calculate the strategy for strategic voting based on the happiness levels
        strategic_voting = self.calculate_strategic_voting(happiness)

        # Calculate the risk associated with the voting strategy
        risk = self.calculate_risk(strategic_voting)

        # Return the analysis results as an Analysis object
        return Analysis(outcome, happiness, total_happiness, strategic_voting, risk)

    def calculate_outcome(self) -> Result:
        """
        Calculate the outcome of a voting system.
        
        Returns:
            Result: The calculated outcome as a Result object.
        """
        # Check different voting scenarios and determine the outcome
        if all(self.voting_scheme[i] == len(self.voting_scheme) - i for i in range(len(self.voting_scheme))):
            return borda_outcome(self.voting_solution)
        
        if self.voting_scheme[0] == 1 and all(self.voting_scheme[i] == 0 for i in range(1, len(self.voting_scheme))):
            return plurality_outcome(self.voting_solution)
        
        if self.voting_scheme[0] == 1 and self.voting_scheme[1] == 1 and all(self.voting_scheme[i] == 0 for i in range(2, len(self.voting_scheme))):
            return for_two_outcome(self.voting_solution)
        
        if all(self.voting_scheme[i] == 1 for i in range(len(self.voting_scheme) - 1)) and self.voting_scheme[-1] == 0:
            return veto_outcome(self.voting_solution)
        
    def calculate_outcome(self) -> Result:
        """
        Calculate the outcome of a voting system.
        
        Returns:
            Result: The calculated outcome as a Result object.
        """
        return self.voting_scheme(self.voting_solution)


    def calculate_happiness(self, outcome: Result) -> HappinessLevel:
        """
        Calculate the happiness level for each voter based on the outcome of the voting system.

        Parameters:
            outcome (Result): The calculated outcome as a Result object.

        Returns:
            HappinessLevel: A Series containing the happiness level for each voter.
        """
        return HappinessLevel(self.voting_solution,outcome)

    def calculate_total_happiness(self, happiness: HappinessLevel) -> float:
        """
        Calculate the total happiness based on the happiness levels of individual voters.

        Parameters:
            happiness (HappinessLevel): A object containing the happiness level for each voter.

        Returns:
            float: The total happiness calculated based on the individual happiness levels.
        """
        return happiness.total
    
    def calculate_strategic_voting(self, happiness: HappinessLevel) -> StrategicVoting:
        """
        Calculate strategic voting behavior based on voters' happiness.

        Parameters:
            happiness (HappinessLevel): Voters' happiness levels.

        Returns:
            StrategicVoting: Object representing strategic voting.
        """
        return StrategicVoting(self.voting_solution).run(happiness, self.voting_scheme)


    def calculate_risk(self, strategic_voting: StrategicVoting) -> float:
        """
        Calculate the risk associated with a particular voting strategy.

        Parameters:
            strategic_voting (StrategicVoting): An object representing strategic voting behavior.

        Returns:
            float: The calculated risk associated with the voting strategy.
        """
        # For simplicity, returning 0.0 for now.
        return strategic_voting.risk

    
    