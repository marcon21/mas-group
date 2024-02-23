from src import outcomes as o
import unittest
import numpy as np
from src.outcomes import Result


class TestOutComes(unittest.TestCase):
    def setUp(self):
        self.s = np.array(
            [
                ["C", "B", "C", "B", "B"],
                ["A", "D", "D", "D", "A"],
                ["D", "C", "A", "C", "D"],
                ["B", "A", "B", "A", "C"],
            ]
        )

    # TODO
    """ 
    def test_plurality_should_resolve_ties(self):
        # Both B and C have max num votes (2)
        self.s[0, :] = ["C", "B", "C", "A", "B"]
        expected_outcome = [("B", 2)]
        outcome = o.plurality_outcome(self.s)

        self.assertCountEqual(outcome, expected_outcome)
        self.assertListEqual(outcome, expected_outcome)
    """

    def test_plurality_should_return_general_expected_outcome(self):
        expected_outcome = Result({"C": 2, "B": 3, "A": 0, "D": 0})
        outcome = o.plurality_outcome(self.s)

        self.assertEqual(outcome.winner, expected_outcome.winner)
        self.assertDictEqual(outcome, expected_outcome)

    def test_for_two_should_return_general_expected_outcome(self):
        expected_outcome = Result({"C": 2, "B": 3, "A": 2, "D": 3})
        outcome = o.for_two_outcome(self.s)

        self.assertEqual(outcome.winner, expected_outcome.winner)
        self.assertDictEqual(outcome, expected_outcome)

    def test_for_veto_should_return_general_expected_outcome(self):
        expected_outcome = Result({"C": 4, "B": 3, "A": 3, "D": 5})
        outcome = o.veto_outcome(self.s)

        self.assertEqual(outcome.winner, expected_outcome.winner)
        self.assertDictEqual(outcome, expected_outcome)

    def test_for_border_count_should_return_general_expected_outcome(self):
        expected_outcome = Result({"C": 8, "B": 9, "A": 5, "D": 8})
        outcome = o.borda_outcome(self.s)

        self.assertEqual(outcome.winner, expected_outcome.winner)
        self.assertDictEqual(outcome, expected_outcome)
