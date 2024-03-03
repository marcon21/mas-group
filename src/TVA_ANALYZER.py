import pandas as pd
import numpy as np
from outcomes import *
import Analysis
from happiness_level import HappinessLevel


class TVA_ANALYZER:
    
    def __init__(self, voting_scheme: np.ndarray, voting_solution: pd.DataFrame):
        """
        Initialize the TVA_ANALYZER with voting scheme and voting solution.

        Parameters:
            voting_scheme (np.ndarray): The NumPy array representing the voting scheme.
            voting_solution (pd.DataFrame): The DataFrame containing the voting solutions, with columns for voter IDs and their preferences.
        """
        # Check the conditions for valid input
        self.condition_checker(voting_scheme, voting_solution)
        
        # Set the attributes
        self.voting_scheme = voting_scheme
        self.voting_solution = voting_solution
    
    @staticmethod
    def condition_checker(voting_scheme: np.ndarray, voting_solution: pd.DataFrame):
        """
        Check conditions for valid input: type check, length check and two candidates condition. Valid voting_scheme is checked later.

        Parameters:
            voting_scheme (np.ndarray): The NumPy array representing the voting scheme.
            voting_solution (pd.DataFrame): The DataFrame containing the voting solutions.
        """
        # Check if voting_solution is a DataFrame
        if not isinstance(voting_solution, pd.DataFrame):
            raise TypeError("voting_solution must be a DataFrame")
        
        # Check if voting_scheme is a NumPy array
        if not isinstance(voting_scheme, np.ndarray):
            raise TypeError("Voting Scheme must be a Np.ndArray")
        
        # Check if the length of voting_scheme matches the number of candidates
        if voting_scheme.shape[0] != voting_solution.shape[1]:
            raise ValueError("Length of voting scheme must be equal to the number of candidates")
        
        # Check if there are at least two candidates
        if voting_solution.shape[1] < 2:
            raise ValueError("There must be at least two candidates")

    def analyze(self) -> 'Analysis':
        """
        Analyze the voting solution and return the analysis results.

        Returns:
            Analysis: An object containing the outcome, happiness, total happiness, strategy, and risk of the voting solution.
        """
        # Calculate the outcome of the voting system
        outcome = self.outcome_calculator()

        # Calculate the happiness level of each voter
        happiness = self.happiness_calculator(outcome)

        # Calculate the total happiness based on individual happiness levels
        total_happiness = self.total_happiness_calculator(happiness)

        # Calculate the strategy for strategic voting
        strategy = self.strategic_voting_calculator()

        # Calculate the risk associated with the voting strategy
        risk = self.risk_calculator(strategy)

        # Return the analysis results
        return Analysis(outcome, happiness, total_happiness, strategy, risk)

    def outcome_calculator(self) -> Result:
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
        
        raise ValueError("Invalid voting scheme")

    def happiness_calculator(self, outcome: Result) -> HappinessLevel:
        """
        Calculate the happiness level for each voter based on the outcome of the voting system.

        Parameters:
            outcome (Result): The calculated outcome as a Result object.

        Returns:
            HappinessLevel: A Series containing the happiness level for each voter.
        """
        return None  # Implement the logic to calculate happiness levels

    def total_happiness_calculator(self, happiness: pd.Series) -> float:
        """
        Calculate the total happiness based on the happiness levels of individual voters.

        Parameters:
            happiness (pd.Series): A Series containing the happiness level for each voter.

        Returns:
            float: The total happiness calculated based on the individual happiness levels.
        """
        # Calculate the total happiness by summing individual happiness levels
        total_happiness = happiness.sum() if not happiness.empty else 0.0
        return total_happiness

    def strategic_voting_calculator(self):
        """
        Calculate strategic voting behavior by comparing the voting scheme with the voting solution.
        """
        # Implementation of strategic voting calculation
        # For simplicity, returning None for now
        return None

    def risk_calculator(self, strategy: pd.DataFrame) -> float:
        """
        Calculate the risk associated with a particular voting strategy.

        Parameters:
            strategy (pd.DataFrame): The DataFrame containing the voting strategy to evaluate.

        Returns:
            float: The calculated risk associated with the strategy.
        """
        # Implementation of risk calculation
        # For simplicity, returning 0.0 for now
        return 0.0