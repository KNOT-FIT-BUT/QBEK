# -*- coding: UTF-8 -*-
""""
Created on 11.04.21

:author:     Martin Dočekal
"""
import copy
import unittest

import torch
from transformers import AutoTokenizer

from qbek.batch import Batch


class TestBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        self.attention_masks = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        self.gt_span_universes = torch.tensor([
            [
                [True, False],
                [False, True]
            ],
            [
                [True, False],
                [False, False]
            ]
        ])

        self.line_offsets = [0, 0]
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.batch = Batch(self.tokens, self.attention_masks, self.gt_span_universes, self.line_offsets,
                           self._tokenizer)

    def test_len(self):
        self.assertEqual(len(self.batch), 2)

    def test_eq(self):
        self.assertEqual(self.batch, self.batch)

    def test_new_diff_class(self):
        self.assertNotEqual(self.batch, "")

    def test_new_diff_tokens(self):
        batch_dif = copy.copy(self.batch)
        batch_dif.tokens = torch.rand(2, 4)
        self.assertNotEqual(self.batch, batch_dif)

    def test_new_diff_attention_masks(self):
        batch_dif = copy.copy(self.batch)
        batch_dif.attention_masks = torch.rand(2, 4)
        self.assertNotEqual(self.batch, batch_dif)

    def test_new_diff_gt_span_universes(self):
        batch_dif = copy.copy(self.batch)
        batch_dif.gt_span_universes = torch.rand(2, 4, 4) > 0.5
        self.assertNotEqual(self.batch, batch_dif)

    def test_new_diff_model_samples(self):
        batch_dif = copy.copy(self.batch)
        batch_dif.line_offsets = [0, 128]
        self.assertNotEqual(self.batch, batch_dif)

    def test_init(self):
        self.assertListEqual(self.tokens.tolist(), self.batch.tokens.tolist())
        self.assertListEqual(self.attention_masks.tolist(), self.batch.attention_masks.tolist())
        self.assertListEqual(self.gt_span_universes.tolist(), self.batch.gt_span_universes.tolist())
        self.assertListEqual(self.line_offsets, self.batch.line_offsets)

    def test_transfer_batch_to_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            cpu_device = torch.device("cpu")

            batch_on_device = self.batch.to(device)

            self.assertEqual(self.batch.tokens.device, cpu_device)
            self.assertEqual(self.batch.attention_masks.device, cpu_device)
            self.assertEqual(self.batch.gt_span_universes.device, cpu_device)

            self.assertEqual(batch_on_device.tokens.device, device)
            self.assertEqual(batch_on_device.attention_masks.device, device)
            self.assertEqual(batch_on_device.gt_span_universes.device, device)

            self.assertListEqual(self.batch.tokens.tolist(), batch_on_device.tokens.tolist())
            self.assertListEqual(self.batch.attention_masks.tolist(), batch_on_device.attention_masks.tolist())
            self.assertListEqual(self.batch.gt_span_universes.tolist(), batch_on_device.gt_span_universes.tolist())
            self.assertListEqual(self.batch.line_offsets, batch_on_device.line_offsets)
            self.assertTrue(self.batch._tokenizer.name_or_path == self._tokenizer.name_or_path
                            == batch_on_device._tokenizer.name_or_path)

            self.batch.gt_span_universes = None

            batch_on_device = self.batch.to(device)
            self.assertIsNone(batch_on_device.gt_span_universes)

        else:
            self.skipTest("Cuda device is not available.")

    def test_split(self):
        batch_first, batch_second = self.batch.split(1)

        self.assertListEqual(batch_first.tokens.tolist(), self.tokens[:1].tolist())
        self.assertListEqual(batch_second.tokens.tolist(), self.tokens[1:].tolist())

        self.assertListEqual(batch_first.attention_masks.tolist(), self.attention_masks[:1].tolist())
        self.assertListEqual(batch_second.attention_masks.tolist(), self.attention_masks[1:].tolist())

        self.assertListEqual(batch_first.gt_span_universes.tolist(), self.gt_span_universes[:1].tolist())
        self.assertListEqual(batch_second.gt_span_universes.tolist(), self.gt_span_universes[1:].tolist())

        self.assertListEqual(batch_first.line_offsets, self.line_offsets[:1])
        self.assertListEqual(batch_second.line_offsets, self.line_offsets[1:])

        # the original should not be changed
        self.assertListEqual(self.tokens.tolist(), self.batch.tokens.tolist())
        self.assertListEqual(self.attention_masks.tolist(), self.batch.attention_masks.tolist())
        self.assertListEqual(self.gt_span_universes.tolist(), self.batch.gt_span_universes.tolist())
        self.assertListEqual(self.line_offsets, self.batch.line_offsets)

        self.assertTrue(batch_first._tokenizer.name_or_path == self._tokenizer.name_or_path
                        == batch_second._tokenizer.name_or_path)


if __name__ == '__main__':
    unittest.main()
