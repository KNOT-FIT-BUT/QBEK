# -*- coding: UTF-8 -*-
""""
Created on 10.04.21

:author:     Martin Dočekal
"""
import math
import unittest
from typing import Optional
from unittest.mock import patch

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from qbek.batch import Batch
from qbek.models.model import Model
from qbek.predictions_filters import SpanSizeFilter, ThresholdFilter


class Base:
    class TestBaseClassForModel(unittest.TestCase):
        """
        Base class for test cases for models.
        """
        def base_set_up(self):
            self.n_best = None
            self.max_span_size = None
            self.score_threshold = None
            self.optimizer_class = AdamW
            self.scheduler_class = None
            self.model: Optional[Model] = None

            self.universe_matrix = torch.tensor([
                [
                    [10, 9, 8, 7, -math.inf, -math.inf],
                    [-math.inf, 6, 5, 4, -math.inf, -math.inf],
                    [-math.inf, -math.inf, 3, 2, -math.inf, -math.inf],
                    [-math.inf, -math.inf, -math.inf, 1, -math.inf, -math.inf],
                    [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                    [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                ],
                [
                    [0, -1, -math.inf, -math.inf, -math.inf, -math.inf],
                    [-math.inf, -2, -math.inf, -math.inf, -math.inf, -math.inf],
                    [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                    [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                    [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                    [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                ]
            ])

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        def test_n_best(self):
            self.assertIsNone(self.model.n_best)
            self.assertEqual(len(self.model._pred_filter.filters), 0)
            self.model.n_best = 3
            self.assertEqual(len(self.model._pred_filter.filters), 0)   # first n filter should not be applied

        def test_max_span_size(self):
            self.assertIsNone(self.model.max_span_size)
            self.assertEqual(len(self.model._pred_filter.filters), 0)
            self.model.max_span_size = 10
            self.assertEqual(len(self.model._pred_filter.filters), 1)
            self.assertTrue(isinstance(self.model._pred_filter.filters[0], SpanSizeFilter))
            self.assertEqual(self.model._pred_filter.filters[0].max_size, 10)

        def test_score_threshold(self):
            self.assertIsNone(self.model.score_threshold)
            self.assertEqual(len(self.model._pred_filter.filters), 0)
            self.model.score_threshold = 0.5
            self.assertEqual(len(self.model._pred_filter.filters), 1)
            self.assertTrue(isinstance(self.model._pred_filter.filters[0], ThresholdFilter))
            self.assertEqual(self.model._pred_filter.filters[0].threshold, 0.5)

        def test_predictions_no_filters(self):
            batch = Batch(torch.tensor([
                [101, 18444, 28088, 70281, 10464, 102, 0],
                [101, 18444, 28088, 70281, 123, 43921, 102]
            ]), torch.empty(2, 4), self.universe_matrix, [0, 128], self.tokenizer)
            res = list(self.model.predictions(self.universe_matrix, batch))
            self.assertListEqual(res,
                                 [
                                     (['key', 'keyph', 'keyphrase', '##ph', '##phrase', '##rase'],
                                      [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
                                     (["key"], [-2])
                                 ])

        def test_predictions_filters(self):
            batch = Batch(torch.tensor([
                [101, 18444, 28088, 70281, 10464, 102, 0],
                [101, 18444, 28088, 70281, 123, 43921, 102]
            ]), torch.empty(2, 4), self.universe_matrix, [0, 128], self.tokenizer)

            def f(*args, **kwargs):
                # simulate
                # self.model.score_threshold = 4
                # self.model.max_span_size = 2
                return torch.tensor([
                    [
                        [10, 9, -math.inf, -math.inf, -math.inf, -math.inf],  # this points to special token, so it should be omitted
                        [-math.inf, 6, 5, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf]
                    ],
                    [
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
                    ]
                ])

            self.model._pred_filter = f
            res = list(self.model.predictions(self.universe_matrix, batch))
            self.assertListEqual(res,
                                 [
                                     (["key", "keyph"], [6.0, 5.0]),
                                     ([], [])
                                 ])

        def test_configure_optimizers(self):
            res = self.model.configure_optimizers()
            self.assertTrue(isinstance(res, self.optimizer_class))

        def test_training_step(self):
            self.model.loss = lambda x, y: 1

            with patch.object(self.model.__class__, 'log', return_value=None) as mock_method:
                self.assertEqual(self.model.training_step(
                    Batch(torch.tensor([
                        [101, 18444, 28088, 70281, 10464, 102, 0],
                        [101, 18444, 28088, 70281, 123, 43921, 102]
                    ]), torch.ones(2, 7), torch.rand(2, 7, 7) > 0.5, [0, 128], self.tokenizer),
                    1
                ), 1)
                self.assertTrue(any(c.args[0] == "train_loss" for c in mock_method.mock_calls))

        def test_predict_span_universe_matrix(self):
            self.model.pred_span_universe_matrix = True

            def mock_use_model(*args, **kwargs):
                return None

            def outputs_2_span_scores(*args, **kwargs):
                return self.universe_matrix

            self.model.use_model = mock_use_model
            self.model.outputs_2_span_scores = outputs_2_span_scores

            batch = Batch(torch.tensor([
                [101, 18444, 28088, 70281, 10464, 102, 0],
                [101, 18444, 28088, 70281, 123, 43921, 102]
            ]), torch.empty(2, 4), self.universe_matrix, [0, 128], self.tokenizer)

            res = self.model.predict_step(batch, 0, None)
            self.assertListEqual(res[0].tolist(), self.universe_matrix.tolist())
            self.assertListEqual(res[1].tokens.tolist(), batch.tokens.tolist())
            self.assertListEqual(res[1].attention_masks.tolist(), batch.attention_masks.tolist())
            self.assertListEqual(res[1].gt_span_universes.tolist(), batch.gt_span_universes.tolist())
            self.assertListEqual(res[1].line_offsets, batch.line_offsets)

        def test_predict(self):
            self.model.pred_span_universe_matrix = False

            def mock_use_model(*args, **kwargs):
                return None

            def outputs_2_span_scores(*args, **kwargs):
                return self.universe_matrix

            def predictions(*args, **kwargs):
                return [(["k1", "k2"], [10, 1]), (["k3", "k4"], [5, 6])]

            self.model.use_model = mock_use_model
            self.model.outputs_2_span_scores = outputs_2_span_scores
            self.model.predictions = predictions

            batch = Batch(torch.tensor([
                [101, 10106, 10410, 10126, 11703, 122, 102],
                [101, 10106, 10410, 10126, 11703, 123, 102]
            ]), torch.empty(2, 4), self.universe_matrix, [0, 0], self.tokenizer)

            res = self.model.predict_step(batch, 0, None)

            self.assertListEqual(res, [
                (["k1", "k2"], [10, 1], 0),
                (["k3", "k4"], [5, 6], 0)
            ])

        def test_validation_step(self):
            self.model.loss = lambda x, y: 1

            with patch.object(self.model.__class__, 'log', return_value=None) as mock_method:
                self.assertEqual(self.model.validation_step(
                    Batch(torch.tensor([
                        [101, 18444, 28088, 70281, 10464, 102, 0],
                        [101, 18444, 28088, 70281, 123, 43921, 102]
                    ]), torch.ones(2, 7), torch.rand(2, 7, 7) > 0.5, [0, 128], self.tokenizer),
                    1
                ), 1)
                self.assertTrue(any(c.args[0] == "val_loss" for c in mock_method.mock_calls))
