# -*- coding: UTF-8 -*-
""""
Created on 15.09.21
Tests for model module.

Tests for Model class are not included as this is just abstract base class that is tested while the classes that
inherits from it are tested.

:author:     Martin Dočekal
"""
import math
import unittest

import torch

from qbek.models.model import OwnBCEWithLogitsLoss


class TestOwnBCEWithLogitsLossBase(unittest.TestCase):

    def setUp(self) -> None:
        self.inputs = torch.tensor([
            [0.5, 0.5, -0.2],
            [0.5, 0.5, -0.2]
        ])
        self.inputs_with_inf = torch.tensor([
            [0.5, -100, -0.2],
            [0.5, 0.5, -0.2]
        ])
        self.targets = torch.tensor([
            [1, 0, 0.0],
            [0, 0, 0]
        ])

        self.targets_no_match = [
            [0, 0, 1.0],
            [0, 0, 0]
        ]

    @staticmethod
    def almost_equal_tensors(x: torch.Tensor, y:torch.Tensor) -> bool:
        return torch.allclose(x, y, atol=1e-04)


class TestOwnBCEWithLogitsLossWithoutWeighting(TestOwnBCEWithLogitsLossBase):

    def setUp(self) -> None:
        super().setUp()
        self.bce = OwnBCEWithLogitsLoss(online_weighting=False, reduction="none")

    def test_base(self):
        expect = torch.tensor([
            [0.4741, 0.9741, 0.5981],
            [0.9741, 0.9741, 0.5981]
        ])
        result = self.bce(self.inputs, self.targets)
        self.assertTrue(self.almost_equal_tensors(expect, result))

    def test_inf_on_input(self):
        expect = torch.tensor([
            [0.4741, 0.0000, 0.5981],
            [0.9741, 0.9741, 0.5981]
        ])
        result = self.bce(self.inputs_with_inf, self.targets)
        self.assertTrue(self.almost_equal_tensors(expect, result))


class TestOwnBCEWithLogitsLossWithMeanReduction(TestOwnBCEWithLogitsLossBase):

    def setUp(self) -> None:
        super().setUp()
        self.bce = OwnBCEWithLogitsLoss(online_weighting=False, reduction="mean")

    def test_base(self):
        result = self.bce(self.inputs, self.targets)
        self.assertAlmostEqual(0.7654, result.item(), places=4)


class TestOwnBCEWithLogitsLossWithWeighting(TestOwnBCEWithLogitsLossBase):

    def setUp(self) -> None:
        super().setUp()
        self.bce = OwnBCEWithLogitsLoss(online_weighting=True, reduction="none")

    def test_base(self):
        expect = torch.tensor([
            [2.3704, 0.9741, 0.5981],
            [0.9741, 0.9741, 0.5981]
        ])
        result = self.bce(self.inputs, self.targets)
        self.assertTrue(self.almost_equal_tensors(expect, result))


if __name__ == '__main__':
    unittest.main()
