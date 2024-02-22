from src import outcomes as o
import unittest
import numpy as np


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

    def test_plurality_should_return_general_expected_outcome(self):
        expected_outcome = [("B", 3)]
        outcome = o.plurality_outcome(self.s)

        self.assertCountEqual(outcome, expected_outcome)
        self.assertListEqual(outcome, expected_outcome)

    def test_plurality_should_resolve_ties(self):
        # Both B and C have max num votes (2)
        self.s[0, :] = ["C", "B", "C", "A", "B"]
        expected_outcome = [("B", 2)]
        outcome = o.plurality_outcome(self.s)

        self.assertCountEqual(outcome, expected_outcome)
        self.assertListEqual(outcome, expected_outcome)

    def test_for_two_should_return_general_expected_outcome(self):
        expected_outcome = [("B", 3), ("D", 3)]
        outcome = o.for_two_outcome(self.s)

        self.assertCountEqual(outcome, expected_outcome)
        self.assertListEqual(outcome, expected_outcome)

    def test_for_two_should_resolve_ties(self):
        # A, B, C have 3 votes
        self.s[0, 0] = "A"
        expected_outcome = [("A", 3), ("B", 3)]
        outcome = o.for_two_outcome(self.s)

        self.assertCountEqual(outcome, expected_outcome)
        self.assertListEqual(outcome, expected_outcome)
