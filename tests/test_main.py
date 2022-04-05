# -*- coding: UTF-8 -*-
""""
Created on 13.09.21
Tests for the main module.

:author:     Martin Dočekal
"""
import os
import unittest
from typing import Optional, Union, Tuple, List

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qbek.__main__ import predict
from qbek.batch import Batch

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/")


class MockModel(LightningModule):

    def __init__(self, pred_span_universe_matrix: bool, predict_n_spans: int = 1, n_best: int = 1):
        super().__init__()
        self._pred_span_universe_matrix = pred_span_universe_matrix
        self.pred_span_universe_matrix_called = False
        self.predict_n_spans = predict_n_spans
        self.n_best = n_best

    def loss(self, outputs, gt_span_universe: torch.Tensor) -> torch.Tensor:
        pass

    def outputs_2_span_scores(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    def use_model(self, batch: Batch) -> Tuple[torch.Tensor, ...]:
        pass

    def eval(self):
        pass

    def to(self, *args, **kwargs):
        return self

    @property
    def pred_span_universe_matrix(self) -> bool:
        self.pred_span_universe_matrix_called = True
        return self._pred_span_universe_matrix

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: Optional[int] = None) -> \
            Union[Tuple[torch.Tensor, Batch], List[Tuple[List[str], List[float], int]]]:

        if self._pred_span_universe_matrix:
            return torch.full((len(batch), 2, 2), float(batch_idx)), batch
        else:
            return [
                (
                    [str(batch_idx)+str(i_s)+str(i) for i in range(self.predict_n_spans)],
                    [batch_idx*100+i_s*10+i for i in range(self.predict_n_spans)],
                    l_o
                ) for i_s, l_o in enumerate(batch.line_offsets)
            ]


class MockModelSamplePerDocument(MockModel):

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: Optional[int] = None) -> \
            Union[Tuple[torch.Tensor, Batch], List[Tuple[List[str], List[float], int]]]:

        if self._pred_span_universe_matrix:
            return torch.full((len(batch), 2, 2), float(batch_idx)), batch
        else:
            return [
                (
                    [str(i_s)+str(l_o)],
                    [l_o],
                    l_o
                ) for i_s, l_o in enumerate(batch.line_offsets)
            ]


class MockDataLoader(DataLoader):

    def __init__(self, length: int, batch_size: int):
        super().__init__([], batch_size)
        self.length = length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(0, len(self)):
            if i % 2:
                line_offsets = [i for _ in range(self.batch_size)]

            else:
                first_part = int(self.batch_size / 2)
                line_offsets = [i for _ in range(first_part)]
                line_offsets += [i+1 for _ in range(self.batch_size - first_part)]

            yield Batch(tokens=torch.full((self.batch_size, 10), i),
                        attention_masks=torch.ones((self.batch_size, 10)),
                        gt_span_universes=torch.triu(torch.ones((self.batch_size, 10, 10))),
                        line_offsets=line_offsets,
                        tokenizer=self.tokenizer)


class MockDataLoaderOneSamplePerDocument(DataLoader):

    def __init__(self, length: int, batch_size: int):
        super().__init__([], batch_size)
        self.length = length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(0, len(self)):
            line_offsets = [i*self.batch_size+x for x in range(self.batch_size)]

            yield Batch(tokens=torch.full((self.batch_size, 10), i),
                        attention_masks=torch.ones((self.batch_size, 10)),
                        gt_span_universes=torch.triu(torch.ones((self.batch_size, 10, 10))),
                        line_offsets=line_offsets,
                        tokenizer=self.tokenizer)


class TestPredict(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 4
        self.data_loader = MockDataLoader(4, self.batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def test_pred(self):
        docs_gt = [
            {
                "spans": ["019", "018"],
                "scores": [19, 18],
                "line_offset": 0
            },
            {
                "spans": ["139", "138"],
                "scores": [139, 138],
                "line_offset": 1
            },
            {
                "spans": ["219", "218"],
                "scores": [219, 218],
                "line_offset": 2
            },
            {
                "spans": ["339", "338"],
                "scores": [339, 338],
                "line_offset": 3
            }
        ]
        for i, doc_prediction in enumerate(
                predict(MockModel(False, predict_n_spans=10, n_best=2), self.data_loader, False, False, False)):
            self.assertListEqual(docs_gt[i]["spans"], doc_prediction.spans)
            self.assertListEqual(docs_gt[i]["scores"], doc_prediction.scores)
            self.assertEqual(docs_gt[i]["line_offset"], doc_prediction.documents_line_offset)

    def test_pred_span_universe_matrix(self):

        # 0     2   2
        # 1     1   6
        # 2     2   2
        # 3     1   6

        matrices = [
            [
                torch.full((2, 2, 2), 0.0)
            ],
            [
                torch.full((2, 2, 2), 0.0),
                torch.full((4, 2, 2), 1.0),
            ],
            [
                torch.full((2, 2, 2), 2.0)
            ],
            [
                torch.full((2, 2, 2), 2.0),
                torch.full((4, 2, 2), 3.0),
            ],
        ]

        batches = [
            [
                Batch(tokens=torch.full((2, 10), 0),
                      attention_masks=torch.ones((2, 10)),
                      gt_span_universes=torch.triu(torch.ones((2, 10, 10))),
                      line_offsets=[0 for _ in range(2)],
                      tokenizer=self.tokenizer)
            ],
            [
                Batch(tokens=torch.full((2, 10), 0),
                      attention_masks=torch.ones((2, 10)),
                      gt_span_universes=torch.triu(torch.ones((2, 10, 10))),
                      line_offsets=[1 for _ in range(2)],
                      tokenizer=self.tokenizer),
                Batch(tokens=torch.full((4, 10), 1),
                      attention_masks=torch.ones((4, 10)),
                      gt_span_universes=torch.triu(torch.ones((4, 10, 10))),
                      line_offsets=[1 for _ in range(4)],
                      tokenizer=self.tokenizer)
            ],
            [
                Batch(tokens=torch.full((2, 10), 2),
                      attention_masks=torch.ones((2, 10)),
                      gt_span_universes=torch.triu(torch.ones((2, 10, 10))),
                      line_offsets=[2 for _ in range(2)],
                      tokenizer=self.tokenizer)
            ],
            [
                Batch(tokens=torch.full((2, 10), 2),
                      attention_masks=torch.ones((2, 10)),
                      gt_span_universes=torch.triu(torch.ones((2, 10, 10))),
                      line_offsets=[3 for _ in range(2)],
                      tokenizer=self.tokenizer),
                Batch(tokens=torch.full((4, 10), 3),
                      attention_masks=torch.ones((4, 10)),
                      gt_span_universes=torch.triu(torch.ones((4, 10, 10))),
                      line_offsets=[3 for _ in range(4)],
                      tokenizer=self.tokenizer)
            ],
        ]

        for i, doc_batches in enumerate(predict(MockModel(True), self.data_loader, False, False, False)):
            self.assertEqual(len(batches[i]), len(doc_batches), msg=f"problem for {i}")
            for b_i, (u_mat, batch) in enumerate(doc_batches):
                self.assertTrue(torch.allclose(matrices[i][b_i], u_mat), msg=f"problem for {i} {b_i}")
                self.assertEqual(batches[i][b_i], batch, msg=f"problem for {i} {b_i}")


class TestPredictOneSamplePerDocument(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 4
        self.data_loader = MockDataLoaderOneSamplePerDocument(2, self.batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def test_pred(self):
        docs_gt = [
            {
                "spans": ["00"],
                "scores": [0],
                "line_offset": 0
            },
            {
                "spans": ["11"],
                "scores": [1],
                "line_offset": 1
            },
            {
                "spans": ["22"],
                "scores": [2],
                "line_offset": 2
            },
            {
                "spans": ["33"],
                "scores": [3],
                "line_offset": 3
            },
            {
                "spans": ["04"],
                "scores": [4],
                "line_offset": 4
            },
            {
                "spans": ["15"],
                "scores": [5],
                "line_offset": 5
            },

            {
                "spans": ["26"],
                "scores": [6],
                "line_offset": 6
            },
            {
                "spans": ["37"],
                "scores": [7],
                "line_offset": 7
            },
        ]

        for i, doc_prediction in enumerate(
                predict(MockModelSamplePerDocument(False, predict_n_spans=10, n_best=2), self.data_loader, False, False, False)):
            self.assertListEqual(docs_gt[i]["spans"], doc_prediction.spans)
            self.assertListEqual(docs_gt[i]["scores"], doc_prediction.scores)
            self.assertEqual(docs_gt[i]["line_offset"], doc_prediction.documents_line_offset)


if __name__ == '__main__':
    unittest.main()
