# -*- coding: UTF-8 -*-
""""
Created on 11.04.21

:author:     Martin Dočekal
"""
import math
import unittest
from typing import Iterable
from unittest.mock import patch

import hyperopt
import numpy as np
import torch
from hyperopt import STATUS_OK
from transformers import AutoTokenizer

from qbek.auto_filter import AutoFilter
from qbek.batch import Batch
from qbek.metrics.extraction_generation import ExtractionGenerationMetric
from qbek.predictions_filters import CompoundFilter, PredictionsFilter, ThresholdFilter, FirstNFilter, \
    SpanSizeFilter
from tests.mocks import MockDumbLemmatizer


class MockFilter(PredictionsFilter):
    """
    This mocked filter is used for testing CompoundFilter.
    It expects on input matrix that is filled with single value (defined in member expect) and adds 1 to this tensor
    and then passes it further.

    So it may be used to check filter cascade in for example this way:

        first filter expects 0 -> returns 1
        second filter expect 1 -> returns 2
        third filter expect 2 -> returns 3
        etc.

    :ivar called: Flag that filter was called (True).
    :vartype called: bool
    :ivar expect: expects on input matrix that is filled with only this value
    :vartype expect: float
    """

    def __init__(self, expect: float, test_case: unittest.TestCase):
        """
        Initialization of mock.

        :param expect: On input expect a tensor filled with only this one expected value.
        :param test_case: Test case of which it is a part.
            Test case To be able to check expected tensor and signalise problems as standard TestCase.
        """
        self.called = False
        self.expect = expect
        self._test_case = test_case

    def __call__(self, span_universe_matrix: torch.Tensor) -> torch.Tensor:
        self.called = True
        self._test_case.assertTrue(torch.allclose(span_universe_matrix,
                                                  torch.full_like(span_universe_matrix, self.expect)))
        return span_universe_matrix + 1


class TestCompoundFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.f_0 = CompoundFilter([])
        self.f_4 = CompoundFilter([MockFilter(i, self) for i in range(4)])

    def test_init(self):
        self.assertEqual(len(self.f_0.filters), 0)
        self.assertEqual(len(self.f_4.filters), 4)

    def test_call(self):
        span_universe_matrix = torch.zeros((3, 4, 4))
        self.assertTrue(torch.allclose(self.f_0(span_universe_matrix), span_universe_matrix))
        self.assertTrue(torch.allclose(self.f_4(span_universe_matrix),
                                       torch.full_like(span_universe_matrix, 4)))


class TestThresholdFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.f = ThresholdFilter(1.0)

    def test_call(self):
        span_universe_matrix = torch.tensor([
            [
                [0.1, 0.2, 0.3],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.0]
            ],
            [
                [0.1, 1.2, 1.3],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.9]
            ]
        ])

        span_universe_matrix_expected = torch.tensor([
            [
                [-math.inf, -math.inf, -math.inf],
                [-math.inf, 1.5, -math.inf],
                [-math.inf, 1.8, 1.0]
            ],
            [
                [-math.inf, 1.2, 1.3],
                [-math.inf, 1.5, -math.inf],
                [-math.inf, 1.8, 1.9]
            ]
        ])

        res = self.f(span_universe_matrix)
        self.assertTrue(torch.allclose(res, span_universe_matrix_expected), msg=res)


class TestFirstNFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.f = FirstNFilter(3)

    def test_call(self):
        span_universe_matrix = torch.tensor([
            [
                [0.1, 0.2, 0.3],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.0]
            ],
            [
                [0.1, 1.2, 1.3],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.9]
            ]
        ])

        span_universe_matrix_expected = torch.tensor([
            [
                [-math.inf, -math.inf, -math.inf],
                [-math.inf, 1.5, -math.inf],
                [-math.inf, 1.8, 1.0]
            ],
            [
                [-math.inf, -math.inf, -math.inf],
                [-math.inf, 1.5, -math.inf],
                [-math.inf, 1.8, 1.9]
            ]
        ])

        res = self.f(span_universe_matrix)
        self.assertTrue(torch.allclose(res, span_universe_matrix_expected), msg=res)

        res = FirstNFilter(100)(span_universe_matrix)
        self.assertTrue(torch.allclose(res, span_universe_matrix), msg=res)


class TestSpanSizeFilter(unittest.TestCase):
    def test_call(self):
        span_universe_matrix = torch.tensor([
            [
                [0.1, 0.2, 0.3],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.0]
            ],
            [
                [0.1, 1.2, 1.3],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.9]
            ]
        ])

        res = SpanSizeFilter(3)(span_universe_matrix)
        self.assertTrue(torch.allclose(res, span_universe_matrix), msg=res)

        span_universe_matrix_expected = torch.tensor([
            [
                [0.1, 0.2, -math.inf],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.0]
            ],
            [
                [0.1, 1.2, -math.inf],
                [0.4, 1.5, 0.6],
                [0.7, 1.8, 1.9]
            ]
        ])
        res = SpanSizeFilter(2)(span_universe_matrix)
        self.assertTrue(torch.allclose(res, span_universe_matrix_expected), msg=res)

        span_universe_matrix_expected = torch.tensor([
            [
                [0.1, -math.inf, -math.inf],
                [0.4, 1.5, -math.inf],
                [0.7, 1.8, 1.0]
            ],
            [
                [0.1, -math.inf, -math.inf],
                [0.4, 1.5, -math.inf],
                [0.7, 1.8, 1.9]
            ]
        ])
        res = SpanSizeFilter(1)(span_universe_matrix)
        self.assertTrue(torch.allclose(res, span_universe_matrix_expected), msg=res)

        span_universe_matrix_expected = torch.tensor([
            [
                [-math.inf, -math.inf, -math.inf],
                [0.4, -math.inf, -math.inf],
                [0.7, 1.8, -math.inf]
            ],
            [
                [-math.inf, -math.inf, -math.inf],
                [0.4, -math.inf, -math.inf],
                [0.7, 1.8, -math.inf]
            ]
        ])
        res = SpanSizeFilter(0)(span_universe_matrix)
        self.assertTrue(torch.allclose(res, span_universe_matrix_expected), msg=res)


class TestAutoFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.all = AutoFilter(MockDumbLemmatizer())
        self.all_but_best = AutoFilter(MockDumbLemmatizer(), n_best=3)
        self.all_but_max_span_size = AutoFilter(MockDumbLemmatizer(), max_span_size=2)
        self.all_but_score_threshold = AutoFilter(MockDumbLemmatizer(), score_threshold=0.5)
        self.nothing = AutoFilter(MockDumbLemmatizer(), n_best=3, max_span_size=2, score_threshold=0.5)

        self.THRESHOLDS_SPACE_SAVED = AutoFilter.THRESHOLDS_SPACE
        AutoFilter.THRESHOLDS_SPACE = 5

        self.predicted_universes = [
            [
                torch.tensor([
                    [
                        [10.0, 5, 3, 1],
                        [1, 15, 11, 1],
                        [1, 2, 15, 1],
                        [1, 1, 1, 1]
                    ],
                    [
                        [10, 5, 3, 1],
                        [1, 11, 2, 1],
                        [1, 2, 11, 1],
                        [1, 1, 1, 1]
                    ]
                ]),
                torch.tensor([
                    [
                        [10.0, 5, 3, 1],
                        [1, 12, 11, 1],
                        [1, 2, 12, 1],
                        [1, 1, 1, 1]
                    ],
                    [
                        [10, 51, 30, 1],
                        [1, 13, 10, 1],
                        [1, 2, 11, 1],
                        [1, 1, 1, 1]
                    ]
                ]),
            ],
            [
                torch.tensor([
                    [
                        [10.0, 5, 3, 1],
                        [1, 15, 10, 1],
                        [1, 2, 15, 1],
                        [1, 1, 1, 1]
                    ],
                    [
                        [10, 51, 30, 1],
                        [1, 20, 10, 1],
                        [1, 2, 21, 1],
                        [1, 1, 1, 1]
                    ]
                ])
            ]
        ]

        self.input_sequences = ["keyphrase1 one", "keyphras2 two", "keyphra3 three", "keyphr4 four", "keyph5 five",
                                "keyp6 six"]

        self.ground_truths = [
            torch.tensor([
                [
                    [False, False, False, False],
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, False, False]
                ],
                [
                    [False, False, False, False],
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False]
                ]
            ]),
            torch.tensor([
                [
                    [False, False, False, False],
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False]
                ],
                [
                    [False, False, False, False],
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False]
                ]
            ]),
            torch.tensor([
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False]
                ],
                [
                    [False, False, False, False],
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False]
                ]
            ])
        ]

        self.docs_ground_truths = [{"key", "##ph"}, {"main", "##ph"}]

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.predictions = [
            [
                (self.predicted_universes[0][0],
                 Batch(torch.tensor([[101, 18444, 28088, 102],
                                     [101, 1000, 1001, 102]]), torch.tensor([]), self.ground_truths[0], [],
                       self.tokenizer)
                 ),
                (self.predicted_universes[0][1],
                 Batch(torch.tensor([[101, 1000, 1001, 102],
                                     [101, 1000, 1001, 102]]), torch.tensor([]), self.ground_truths[1], [],
                       self.tokenizer)
                 )
            ],
            [
                (self.predicted_universes[1][0],
                 Batch(torch.tensor([[101, 1000, 1001, 102],
                                     [101, 12126, 28088, 102]]), torch.tensor([]), self.ground_truths[2], [],
                       self.tokenizer))
            ],
        ]

    def tearDown(self) -> None:
        AutoFilter.THRESHOLDS_SPACE = self.THRESHOLDS_SPACE_SAVED

    def test_all_params_known(self):
        self.assertFalse(self.all.all_params_known())
        self.assertFalse(self.all_but_best.all_params_known())
        self.assertFalse(self.all_but_max_span_size.all_params_known())
        self.assertFalse(self.all_but_score_threshold.all_params_known())
        self.assertTrue(self.nothing.all_params_known())

    def test_call(self):
        class MockFmin:
            def __init__(self):
                self.n_best = None
                self.max_span_size = None
                self.score_threshold = None
                self.called = False

            def __call__(self, wrapper, space, algo, max_evals, trials: hyperopt.Trials):
                self.called = True
                self.n_best = space["n_best"]
                self.max_span_size = space["max_span_size"]
                self.score_threshold = space["score_threshold"]
                trials._trials = [
                    {
                        "result": {
                            "loss": 0.0,
                            "status": STATUS_OK,
                            "config": {"max_span_size": 1.0, "score_threshold": 2.0, "n_best": 3.0}
                        }
                    }
                ]
        with patch('hyperopt.hp.choice', lambda name, x: x), patch("hyperopt.hp.uniform", lambda name, mi, ma: [mi, ma]):
            with patch('hyperopt.fmin', MockFmin()) as mock:
                self.all(self.predictions, self.docs_ground_truths)
                self.assertEqual(mock.n_best, [2])
                self.assertEqual(mock.max_span_size, [1])
                self.assertEqual(mock.score_threshold, [1, 51])

            with patch('hyperopt.fmin', MockFmin()) as mock:
                self.all_but_best(self.predictions, self.docs_ground_truths)
                self.assertEqual(mock.n_best, [3])
                self.assertEqual(mock.max_span_size, [1])
                self.assertEqual(mock.score_threshold, [1, 51])

            with patch('hyperopt.fmin', MockFmin()) as mock:
                self.all_but_max_span_size(self.predictions, self.docs_ground_truths)
                self.assertEqual(mock.n_best, [2])
                self.assertEqual(mock.max_span_size, [2])
                self.assertEqual(mock.score_threshold, [1, 51])

            with patch('hyperopt.fmin', MockFmin()) as mock:
                self.all_but_score_threshold(self.predictions, self.docs_ground_truths)
                self.assertFalse(mock.called)   # the grid search should be used

            with patch('hyperopt.fmin', MockFmin()) as mock:
                self.nothing(self.predictions, self.docs_ground_truths)
                self.assertIsNone(mock.n_best)
                self.assertIsNone(mock.max_span_size)
                self.assertIsNone(mock.score_threshold)

    def test_spans(self):
        universe_matrix = np.array([
            [-math.inf, 0, -math.inf],
            [-math.inf, 0, -math.inf],
            [-math.inf, -math.inf, -math.inf]
        ])

        self.assertListEqual(AutoFilter.spans(universe_matrix, -math.inf),
                             [(0, 0, 2), (0, 1, 2)])

    def test_eval_mock(self):
        ground_truths = [["keyphrase1", "one", "keyphras2", "keyphra3", "keyphr4"], ["keyph5", "keyp6"]]

        class MockMetric:
            def __init__(self):
                self.evaluated_predictions = None

            def eval(self, spans: Iterable[Iterable[str]], gt_spans: Iterable[Iterable[str]]):
                self.evaluated_predictions = spans
                return {t: (1.0, 1.0, 1.0) for t in ExtractionGenerationMetric.EVALUATION_TYPE_NAMES}

        self.all.metric = MockMetric()
        self.all._predictions = self.predicted_universes
        self.all._ground_truths = ground_truths
        self.all._batches = [[batch for _, batch in doc] for doc in self.predictions]

        self.all.eval((1, 10.0, 2))
        self.assertListEqual([set(doc_pred) for doc_pred in self.all.metric.evaluated_predictions],
                             self.docs_ground_truths)

    def test_eval(self):
        self.all._predictions = self.predicted_universes
        self.all._ground_truths = self.docs_ground_truths
        self.all._batches = [[batch for _, batch in doc] for doc in self.predictions]
        self.assertEqual(1.0, self.all.eval((1, 10.0, 2)))


if __name__ == '__main__':
    unittest.main()
