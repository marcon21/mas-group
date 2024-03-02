import pandas as pd
import numpy as np
import Analysis as Analysis
from outcomes import *
from happiness_level import HappinessLevel
from strategic_voting_risk import StrategicVoting
from utils import VotingArray
import voting_scenario_generator as vsg  # Import your scenario generator

class TVA_ANALYZER:
    
    def __init__(self, voting_scheme: np.ndarray, voting_preferences: VotingArray):
        """
        Initialize the TVA_ANALYZER with voting scheme and voting preferences.

        Parameters:
            voting_scheme (np.ndarray): The NumPy array representing the voting scheme.
            voting_preferences (VotingArray): The VotingArray containing the voting preferencess, with columns for voter IDs and their preferences.
        """
        # Check the conditions for valid input
        self.condition_checker(voting_scheme, voting_preferences)
        
        # Set the attributes
        self.voting_scheme = voting_scheme
        self.voting_preferences = voting_preferences
    
    @staticmethod
    def condition_checker(voting_scheme: np.ndarray, voting_preferences: VotingArray):
        """
        Check conditions for valid input: type check, length check and two candidates condition. Valid voting_scheme is checked later.

        Parameters:
            voting_scheme (np.ndarray): The NumPy array representing the voting scheme.
            voting_preferences (VotingArray): The VotingArray containing the voting preferencess.
        """
        # Check if voting_preferences is a VotingArray
        if not isinstance(voting_preferences, VotingArray):
            raise TypeError("voting_preferences must be a VotingArray")
        
        # Check if voting_scheme is a NumPy array
        if not isinstance(voting_scheme, np.ndarray):
            raise TypeError("Voting Scheme must be a Np.ndArray")
        
        # Check if the length of voting_scheme matches the number of candidates
        if voting_scheme.shape[0] != voting_preferences.shape[1]:
            raise ValueError(f"Length of voting scheme must match number of preferences. Found {voting_scheme.shape[0]} votes in the scheme and {voting_preferences.shape[1]} preferences per voter.")

        # Check if there are at least two candidates
        if voting_preferences.shape[1] < 2:
            raise ValueError("There must be at least two candidates")

    def analyze(self) -> 'Analysis':
        """
        Analyze the voting preferences and return the analysis results.

        Returns:
            Analysis: An object containing the outcome, happiness, total happiness, strategy, and risk of the voting preferences.
        """
        # Calculate the outcome of the voting system
        outcome = self.outcome_calculator()

         # Create an instance of HappinessLevel based on initial outcome
        happiness = HappinessLevel(self.voting_preferences, outcome.winner)

        # Calculate the total happiness based on individual happiness levels
        total_happiness = happiness.total

        # Instantiate and run StrategicVoting
        strategic_voting = StrategicVoting(self.voting_preferences)
        strategic_analysis = strategic_voting.run(happiness, plurality_outcome)  # Pass the function used to calculate outcomes

        # Extract the strategic voting strategy and risk calculated by StrategicVoting
        strategy = strategic_analysis.best  # Assuming 'best' holds the best strategic voting for each voter
        risk = strategic_analysis.risk  # Assuming 'risk' holds the calculated risk from strategic voting



        # Return the analysis results
        analysis = Analysis.Analysis(outcome, happiness, total_happiness, strategy, risk)


        return analysis

    def outcome_calculator(self) -> Result:
        """
        Calculate the outcome of a voting system.
        
        Returns:
            Result: The calculated outcome as a Result object.
        """
        # Check different voting scenarios and determine the outcome
        if all(self.voting_scheme[i] == len(self.voting_scheme) - i for i in range(len(self.voting_scheme))):
            return borda_outcome(self.voting_preferences)
        
        if self.voting_scheme[0] == 1 and all(self.voting_scheme[i] == 0 for i in range(1, len(self.voting_scheme))):
            return plurality_outcome(self.voting_preferences)
        
        if self.voting_scheme[0] == 1 and self.voting_scheme[1] == 1 and all(self.voting_scheme[i] == 0 for i in range(2, len(self.voting_scheme))):
            return for_two_outcome(self.voting_preferences)
        
        if all(self.voting_scheme[i] == 1 for i in range(len(self.voting_scheme) - 1)) and self.voting_scheme[-1] == 0:
            return veto_outcome(self.voting_preferences)
        
        raise ValueError("Invalid voting scheme")


if __name__ == "__main__":
    import utils as utils

    # Generate the voting scenarios first
    num_examples, num_preferences = vsg.main()  # Call the generator which will prompt user inputs and generate the examples

    for i in range(1, num_examples + 1):
        table_name = f"voting{i}"
        print(f"\nAnalyzing {table_name}...")

        # Use the existing read_voting function to fetch the specific voting table
        voting_table = utils.read_voting("input/voting_result.json", table_name=table_name)

        # Define voting_scheme based on the requirements or data
        voting_scheme = np.array([1, 0, 0] + [0] * (len(voting_table[0]) - 3))

        analyzer = TVA_ANALYZER(voting_scheme, voting_table)
        analysis_results = analyzer.analyze()

        print(analysis_results)

