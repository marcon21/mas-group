class Analysis:
    def __init__(self, outcome: Result, happiness: HappinessLevel, total_happiness: float, strategies: np.ndarray, risk: float):
        """
        Initialize an Analysis object with the provided parameters.

        Parameters:
            outcome (float): The outcome of the voting analysis.
            happiness (pd.Series): A Series containing the happiness level for each voter.
            total_happiness (float): The total happiness calculated based on the individual happiness levels.
            strategies (list[pd.DataFrame]): A list of DataFrames containing information about strategic voting behavior.
            risk (int): The calculated risk associated with the voting analysis.
        """
        self.outcome = outcome
        self.happiness = happiness
        self.total_happiness = total_happiness
        self.strategies = strategies
        self.risk = risk
