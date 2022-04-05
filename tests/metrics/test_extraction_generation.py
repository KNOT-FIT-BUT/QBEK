# -*- coding: UTF-8 -*-
""""
Created on 13.09.21
Tests for ExtractionGenerationMetric.
:author:     Martin Dočekal
"""
import unittest
from typing import Dict, Tuple

from qbek.metrics.extraction_generation import ExtractionGenerationMetric
from qbek.tokenizer import AlphaTokenizer
from tests.mocks import MockLastCapLemmatizer


class TestExtractionGenerationMetric(unittest.TestCase):

    def setUp(self) -> None:
        self.metric = ExtractionGenerationMetric(AlphaTokenizer(), MockLastCapLemmatizer())

    def compare_results(self, expected: Dict[str, Tuple[float, float, float]],
                        predicted: Dict[str, Tuple[float, float, float]]):
        self.assertEqual(expected.keys(), predicted.keys(), msg="Different evaluation types.")

        for type_name, results in expected.items():
            self.assertAlmostEqual(results[0], predicted[type_name][0], msg=f"{type_name}: precision")
            self.assertAlmostEqual(results[1], predicted[type_name][1], msg=f"{type_name}: recall")
            self.assertAlmostEqual(results[2], predicted[type_name][2], msg=f"{type_name}: f1")

    def test_eval_all_empty(self):
        gt_spans = [[], [], [], []]
        pred_spans = [[], [], [], []]

        expected = {
            "exact": (1.0, 1.0, 1.0),
            "lemma": (1.0, 1.0, 1.0),
            "lower case": (1.0, 1.0, 1.0),
            "lower case lemma": (1.0, 1.0, 1.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval_all_no_predictions(self):
        gt_spans = [["kA", "kB"], [], ["kD", "kE"], ["kF", "kG", "kH"]]
        pred_spans = [[], [], [], []]

        expected = {
            "exact": (0.0, 0.0, 0.0),
            "lemma": (0.0, 0.0, 0.0),
            "lower case": (0.0, 0.0, 0.0),
            "lower case lemma": (0.0, 0.0, 0.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval_all_no_grund_truths(self):
        gt_spans = [[], [], [], []]
        pred_spans = [["kA", "kB"], [], ["kD", "kE"], ["kF", "kG", "kH"]]

        expected = {
            "exact": (0.0, 0.0, 0.0),
            "lemma": (0.0, 0.0, 0.0),
            "lower case": (0.0, 0.0, 0.0),
            "lower case lemma": (0.0, 0.0, 0.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval_oracle_with_empty(self):
        gt_spans = [["kA", "kB"], [], ["kD", "kE"], ["kF", "kG", "kH"]]
        pred_spans = [["kA", "kB"], [], ["kD", "kE"], ["kF", "kG", "kH"]]

        expected = {
            "exact": (1.0, 1.0, 1.0),
            "lemma": (1.0, 1.0, 1.0),
            "lower case": (1.0, 1.0, 1.0),
            "lower case lemma": (1.0, 1.0, 1.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval_oracle(self):
        gt_spans = [["kA", "kB"], ["kC"], ["kD", "kE"], ["kF", "kG", "kH"]]
        pred_spans = [["kA", "kB"], ["kC"], ["kD", "kE"], ["kF", "kG", "kH"]]

        expected = {
            "exact": (1.0, 1.0, 1.0),
            "lemma": (1.0, 1.0, 1.0),
            "lower case": (1.0, 1.0, 1.0),
            "lower case lemma": (1.0, 1.0, 1.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval(self):
        gt_spans = [["aA", "bB"], ["cC"], ["dD", "eE"], ["fF", "gG", "hH"]]
        pred_spans = [["aA"], ["cC"], ["dD", "eE"], ["hH", "aA"]]

        # correct = 5
        # extracted = 6
        # ground truth = 8

        res = (5/6, 5/8, (2*(5/8)*(5/6))/(5/8+5/6))
        expected = {
            "exact": res,
            "lemma": res,
            "lower case": res,
            "lower case lemma": res
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval_lemma(self):
        gt_spans = [["aA", "bB"], ["cC"], ["dD", "eE"], ["fF", "gG", "hH"]]
        pred_spans = [["a", "b"], ["c"], ["d", "e"], ["f", "g", "h"]]

        expected = {
            "exact": (0.0, 0.0, 0.0),
            "lemma": (1.0, 1.0, 1.0),
            "lower case": (0.0, 0.0, 0.0),
            "lower case lemma": (1.0, 1.0, 1.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval_lower_case(self):
        gt_spans = [["kA", "kB"], ["kC"], ["kD", "kE"], ["kF", "kG", "kH"]]
        pred_spans = [["KA", "KB"], ["KC"], ["KD", "KE"], ["KF", "KG", "KH"]]

        expected = {
            "exact": (0.0, 0.0, 0.0),
            "lemma": (0.0, 0.0, 0.0),
            "lower case": (1.0, 1.0, 1.0),
            "lower case lemma": (1.0, 1.0, 1.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))

    def test_eval_lower_case_lemma(self):
        gt_spans = [["aA", "aB", "bB"], ["cC"], ["dD", "eR"], ["fF", "gG", "hH"]]
        pred_spans = [["A", "B"], ["C"], ["D", "E"], ["F", "G", "H"]]

        expected = {
            "exact": (0.0, 0.0, 0.0),
            "lemma": (0.0, 0.0, 0.0),
            "lower case": (0.0, 0.0, 0.0),
            "lower case lemma": (1.0, 1.0, 1.0)
        }

        self.compare_results(expected, self.metric.eval(pred_spans, gt_spans))


if __name__ == '__main__':
    unittest.main()
