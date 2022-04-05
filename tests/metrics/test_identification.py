# -*- coding: UTF-8 -*-
""""
Created on 10.04.21

:author:     Martin Dočekal
"""

import unittest

from qbek.metrics.identification import MultiLabelIdentificationMetrics


class TestMultiLabelIdentificationMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = MultiLabelIdentificationMetrics()

    def test_eval_default(self):
        spans = [[(1, 3), (4, 6)], [(2, 4), (5, 7)]]
        res = self.metric.eval(spans, spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 1.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 1.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 1.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 1.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 1.0)
        self.assertEqual(res["partOfAny"]["precision"], 1.0)
        self.assertEqual(res["partOfAny"]["recall"], 1.0)
        self.assertEqual(res["partOfAny"]["f1"], 1.0)

        self.assertEqual(res["includesAny"]["accuracy"], 1.0)
        self.assertEqual(res["includesAny"]["precision"], 1.0)
        self.assertEqual(res["includesAny"]["recall"], 1.0)
        self.assertEqual(res["includesAny"]["f1"], 1.0)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 1.0)

        self.assertEqual(res["noMatchAtAll"], 0)

        gt_spans = [[(1, 3), (4, 6), (10, 12), (20, 22)], [(2, 4), (5, 7), (13, 14), (23, 25)]]

        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.5)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 1.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.5)
        self.assertAlmostEqual(res["exactMatchWithAny"]["f1"], 0.66666666666)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.5)
        self.assertEqual(res["partOfAny"]["precision"], 1.0)
        self.assertEqual(res["partOfAny"]["recall"], 0.5)
        self.assertAlmostEqual(res["partOfAny"]["f1"], 0.66666666666)

        self.assertEqual(res["includesAny"]["accuracy"], 0.5)
        self.assertEqual(res["includesAny"]["precision"], 1.0)
        self.assertEqual(res["includesAny"]["recall"], 0.5)
        self.assertAlmostEqual(res["includesAny"]["f1"], 0.66666666666)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 0.5)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.5)
        self.assertAlmostEqual(res["sharedPartOfSpan"]["f1"], 0.66666666666)

        self.assertEqual(res["noMatchAtAll"], 0)

        gt_spans = [
            [(100, 201), (400, 501), (1000, 1101), (2000, 2101)],
            [(200, 301), (500, 601), (1300, 1301), (2300, 2401)]
        ]
        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.0)
        self.assertEqual(res["partOfAny"]["precision"], 0.0)
        self.assertEqual(res["partOfAny"]["recall"], 0.0)
        self.assertEqual(res["partOfAny"]["f1"], 0.0)

        self.assertEqual(res["includesAny"]["accuracy"], 0.0)
        self.assertEqual(res["includesAny"]["precision"], 0.0)
        self.assertEqual(res["includesAny"]["recall"], 0.0)
        self.assertEqual(res["includesAny"]["f1"], 0.0)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 0.0)

        self.assertEqual(res["noMatchAtAll"], 2)

        # only some are part of any known
        spans = [[(1, 4), (2, 4)]]

        gt_spans = [[(0, 2), (0, 4)]]

        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 1.0)
        self.assertEqual(res["partOfAny"]["precision"], 1.0)
        self.assertEqual(res["partOfAny"]["recall"], 1.0)
        self.assertEqual(res["partOfAny"]["f1"], 1.0)

        self.assertEqual(res["includesAny"]["accuracy"], 0.0)
        self.assertEqual(res["includesAny"]["precision"], 0.0)
        self.assertEqual(res["includesAny"]["recall"], 0.0)
        self.assertEqual(res["includesAny"]["f1"], 0.0)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 1.0)

        self.assertEqual(res["noMatchAtAll"], 0)

        spans = [[(1, 3), (4, 6)], [(2, 4), (5, 7)]]
        gt_spans = [[(1, 4), (400, 501), (1000, 1101), (2000, 2101)], [(0, 11), (500, 601), (1300, 1301), (2300, 2401)]]

        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.35)
        self.assertEqual(res["partOfAny"]["precision"], 0.75)
        self.assertEqual(res["partOfAny"]["recall"], 0.375)
        self.assertEqual(res["partOfAny"]["f1"], 0.5)

        self.assertEqual(res["includesAny"]["accuracy"], 0.0)
        self.assertEqual(res["includesAny"]["precision"], 0.0)
        self.assertEqual(res["includesAny"]["recall"], 0.0)
        self.assertEqual(res["includesAny"]["f1"], 0.0)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 0.35)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.75)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.375)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 0.5)

        self.assertEqual(res["noMatchAtAll"], 0)

        # some includes any known
        spans = [[(0, 1), (0, 6)], [(11, 12), (11, 13)]]
        gt_spans = [[(1, 4)], [(0, 11)]]

        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.0)
        self.assertEqual(res["partOfAny"]["precision"], 0.0)
        self.assertEqual(res["partOfAny"]["recall"], 0.0)
        self.assertEqual(res["partOfAny"]["f1"], 0.0)

        self.assertEqual(res["includesAny"]["accuracy"], 0.25)
        self.assertEqual(res["includesAny"]["precision"], 0.25)
        self.assertEqual(res["includesAny"]["recall"], 0.5)
        self.assertAlmostEqual(res["includesAny"]["f1"], 0.3333333333333)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 0.25)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.25)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.5)
        self.assertAlmostEqual(res["sharedPartOfSpan"]["f1"], 0.3333333333333)

        self.assertEqual(res["noMatchAtAll"], 1)

        # shared part with any of known

        spans = [[(1, 3), (4, 6)], [(2, 4), (5, 7)]]
        gt_spans = [[(0, 2), (6, 8)], [(4, 6), (0, 2)]]

        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.0)
        self.assertEqual(res["partOfAny"]["precision"], 0.0)
        self.assertEqual(res["partOfAny"]["recall"], 0.0)
        self.assertEqual(res["partOfAny"]["f1"], 0.0)

        self.assertEqual(res["includesAny"]["accuracy"], 0.0)
        self.assertEqual(res["includesAny"]["precision"], 0.0)
        self.assertEqual(res["includesAny"]["recall"], 0.0)
        self.assertAlmostEqual(res["includesAny"]["f1"], 0.0)

        self.assertAlmostEqual(res["sharedPartOfSpan"]["accuracy"], 0.3333333333333)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.5)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.5)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 0.5)

        self.assertEqual(res["noMatchAtAll"], 0)

        # mixed

        spans = [[(1, 3), (4, 6), (10, 13), (20, 22), (0, 1)], []]
        gt_spans = [[(1, 3), (4, 7), (11, 13), (19, 21)], [(1, 3), (5, 8), (14, 16), (32, 34)]]

        res = self.metric.eval(spans, gt_spans)

        self.assertAlmostEqual(res["exactMatchWithAny"]["accuracy"], 0.0625)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.1)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.125)
        self.assertAlmostEqual(res["exactMatchWithAny"]["f1"], 0.111111111)

        self.assertAlmostEqual(res["partOfAny"]["accuracy"], 0.14285714285714285)
        self.assertEqual(res["partOfAny"]["precision"], 0.2)
        self.assertEqual(res["partOfAny"]["recall"], 0.25)
        self.assertAlmostEqual(res["partOfAny"]["f1"], 0.222222222)

        self.assertEqual(res["includesAny"]["accuracy"], 0.14285714285714285)
        self.assertEqual(res["includesAny"]["precision"], 0.2)
        self.assertEqual(res["includesAny"]["recall"], 0.25)
        self.assertAlmostEqual(res["includesAny"]["f1"], 0.222222222)

        self.assertAlmostEqual(res["sharedPartOfSpan"]["accuracy"], 0.4)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.4)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.5)
        self.assertAlmostEqual(res["sharedPartOfSpan"]["f1"], 0.444444444)

        self.assertEqual(res["noMatchAtAll"], 1)

        # empty

        spans = [[], []]
        gt_spans = [[], []]
        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 1.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 1.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 1.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 1.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 1.0)
        self.assertEqual(res["partOfAny"]["precision"], 1.0)
        self.assertEqual(res["partOfAny"]["recall"], 1.0)
        self.assertEqual(res["partOfAny"]["f1"], 1.0)

        self.assertEqual(res["includesAny"]["accuracy"], 1.0)
        self.assertEqual(res["includesAny"]["precision"], 1.0)
        self.assertEqual(res["includesAny"]["recall"], 1.0)
        self.assertEqual(res["includesAny"]["f1"], 1.0)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 1.0)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 1.0)

        self.assertEqual(res["noMatchAtAll"], 0)

        spans = [[(0, 2)],[]]
        gt_spans = [[], []]

        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.5)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.5)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.5)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.5)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.5)
        self.assertEqual(res["partOfAny"]["precision"], 0.5)
        self.assertEqual(res["partOfAny"]["recall"], 0.5)
        self.assertEqual(res["partOfAny"]["f1"], 0.5)

        self.assertEqual(res["includesAny"]["accuracy"], 0.5)
        self.assertEqual(res["includesAny"]["precision"], 0.5)
        self.assertEqual(res["includesAny"]["recall"], 0.5)
        self.assertEqual(res["includesAny"]["f1"], 0.5)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 0.5)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.5)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.5)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 0.5)

        self.assertEqual(res["noMatchAtAll"], 1)

        spans = [[(0, 2)], [(2, 4)]]
        gt_spans = [[], []]
        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.0)
        self.assertEqual(res["partOfAny"]["precision"], 0.0)
        self.assertEqual(res["partOfAny"]["recall"], 0.0)
        self.assertEqual(res["partOfAny"]["f1"], 0.0)

        self.assertEqual(res["includesAny"]["accuracy"], 0.0)
        self.assertEqual(res["includesAny"]["precision"], 0.0)
        self.assertEqual(res["includesAny"]["recall"], 0.0)
        self.assertEqual(res["includesAny"]["f1"], 0.0)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 0.0)

        self.assertEqual(res["noMatchAtAll"], 2)

    def test_eval_edge_cases_f1(self):
        spans = [[(2, 3)], [(9, 20)]]
        gt_spans = [[(1, 2)], [(8, 9)]]
        res = self.metric.eval(spans, gt_spans)

        self.assertEqual(res["exactMatchWithAny"]["accuracy"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["precision"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["recall"], 0.0)
        self.assertEqual(res["exactMatchWithAny"]["f1"], 0.0)

        self.assertEqual(res["partOfAny"]["accuracy"], 0.0)
        self.assertEqual(res["partOfAny"]["precision"], 0.0)
        self.assertEqual(res["partOfAny"]["recall"], 0.0)
        self.assertEqual(res["partOfAny"]["f1"], 0.0)

        self.assertEqual(res["includesAny"]["accuracy"], 0.0)
        self.assertEqual(res["includesAny"]["precision"], 0.0)
        self.assertEqual(res["includesAny"]["recall"], 0.0)
        self.assertEqual(res["includesAny"]["f1"], 0.0)

        self.assertEqual(res["sharedPartOfSpan"]["accuracy"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["precision"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["recall"], 0.0)
        self.assertEqual(res["sharedPartOfSpan"]["f1"], 0.0)

        self.assertEqual(res["noMatchAtAll"], 2)


if __name__ == '__main__':
    unittest.main()
