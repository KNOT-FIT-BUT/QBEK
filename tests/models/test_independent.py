# -*- coding: UTF-8 -*-
""""
Created on 10.04.21

:author:     Martin Dočekal
"""
import math
import unittest

import torch
from torch.optim import AdamW

from qbek.batch import Batch
from qbek.models.independent import Independent
from qbek.training.optimizer_factory import AnyOptimizerFactory
from tests.mocks import MockTransformer, MockForward
from tests.models.model_base import Base


class TestIndependent(Base.TestBaseClassForModel):

    def setUp(self):
        self.base_set_up()
        self.model = Independent("bert-base-multilingual-cased",
                                 optimizer=AnyOptimizerFactory(AdamW, attr={"lr": 0.5, "weight_decay": 1e-2}))

    def test_loss(self):
        outputs_starts = torch.tensor([
            [100, 10, 1.0],
            [10, 100, 1.0]
        ])
        outputs_ends = torch.tensor([
            [1.0, 100, 10.],
            [10., 100, 1]
        ])

        gt_span_universe = torch.tensor([
            [
                [False, True, False],
                [False, False, False],
                [False, False, False]
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False]
            ]
        ])

        gt_starts = torch.tensor([
            [1, 0, 0.0],
            [0, 0, 0]
        ])
        gt_ends = torch.tensor([
            [0, 1.0, 0],
            [0, 0, 0]
        ])

        loss = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor(5))(outputs_starts, gt_starts) + \
               torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor(5))(outputs_ends, gt_ends)
        self.assertAlmostEqual(self.model.loss((outputs_starts, outputs_ends), gt_span_universe).item(), loss.item())

    def test_outputs_2_span_scores(self):
        starts_scores = torch.tensor([
            [-10.0, 10, 0],
            [-10, 10, -10]
        ])

        ends_scores = torch.tensor([
            [-10.0, -10, 10],
            [-10, 10, -10]
        ])

        attention_mask = torch.tensor([
            [1.0, 1, 1],
            [1, 1, 0]
        ])

        res = self.model.outputs_2_span_scores((starts_scores, ends_scores), attention_mask)

        res_expected = torch.tensor([
            [
                [-2.0000e+01, -2.0000e+01, -1.0000e+01],
                [-math.inf, -1.0000e+01, -9.0836e-05],
                [-math.inf, -math.inf, -6.9320e-01]
            ],
            [
                [-2.0000e+01, -1.0000e+01, -math.inf],
                [-math.inf, -9.0836e-05, -math.inf],
                [-math.inf, -math.inf, -math.inf]
            ]
        ])
        self.assertTrue(torch.allclose(res, res_expected, atol=1e-6))

    def test_use_model(self):
        batch = Batch(torch.rand(2, 4), torch.empty(2, 4).random_(2), None, [], self.tokenizer)
        forward_mock = MockForward(batch.tokens, batch.attention_masks, (torch.rand(2, 4), torch.rand(2, 4)), None,
                                   self)
        self.model.forward = forward_mock

        res = self.model.use_model(batch)
        self.assertEqual(res[0].tolist(), forward_mock.expected_output[0].tolist())
        self.assertEqual(res[1].tolist(), forward_mock.expected_output[1].tolist())
        self.assertTrue(forward_mock.called)

    @torch.no_grad()
    def test_forward(self):
        input_ids, attention_mask = torch.rand(2, 3), torch.empty(2, 3).random_(2)

        trans_out = torch.tensor([
            [[0.8626, 0.2533],
             [0.2098, 0.0269],
             [0.3106, 0.3042]],
            [[0.5921, 0.4335],
             [0.9740, 0.3508],
             [0.5757, 0.1870]]
        ])

        transformer_mock = MockTransformer(10, self, trans_out)
        self.model.start_end_projection = torch.nn.Linear(2, 2, bias=False)
        self.model.start_end_projection.weight[0, :] = 1
        self.model.start_end_projection.weight[1, :] = 1 / 2

        self.model.transformer = transformer_mock

        start, ends = self.model(input_ids, attention_mask)
        self.assertTrue(torch.allclose(start,
                                       torch.tensor([[1.1159000396728516, 0.23669999837875366, 0.614799976348877],
                                                     [1.0255999565124512, 1.3248000144958496, 0.7626999616622925]]),
                                       atol=1e-6))
        self.assertTrue(torch.allclose(ends,
                                       torch.tensor([[0.5579500198364258, 0.11834999918937683, 0.3073999881744385],
                                                     [0.5127999782562256, 0.6624000072479248, 0.38134998083114624]]),
                                       atol=1e-6))

    @torch.no_grad()
    def test_forward_shapes(self):
        # controls just expected output shapes but on the other hand uses real transformer

        input_ids = torch.tensor([
            [100, 256, 300, 101],
            [100, 200, 101, 0]
        ])

        attention_mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0]
        ])

        start, ends = self.model(input_ids, attention_mask)

        self.assertEqual(start.shape, input_ids.shape)
        self.assertEqual(ends.shape, input_ids.shape)


if __name__ == '__main__':
    unittest.main()
