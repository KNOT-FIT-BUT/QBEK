# -*- coding: UTF-8 -*-
""""
Created on 30.09.21
Tests for preprocessing module.

:author:     Martin Dočekal
"""
import copy
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from qbek.datasets.czech_library import CzechLibrarySampleSplitter
from qbek.datasets.preprocessing import PreprocessedDataset, SampleConvertor

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

TEST_PATH = os.path.join(SCRIPT_PATH, "fixtures/test.txt")

DATASET_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset.txt")
PREP_DATASET_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset.prep")
PREP_DATASET_PATH_TMP = os.path.join(SCRIPT_PATH, "tmp/dataset.prep")
PREP_NO_INDEX_DATASET_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset_no_index.prep")
PREP_DIFF_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset_diff_prep.prep")

DATASET_EMBED_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset_embed.txt")
PREP_DATASET_EMBED_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset_embed.prep")

DATASET_TITLE_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset_title.txt")
PREP_DATASET_TITLE_PATH = os.path.join(SCRIPT_PATH, "fixtures/dataset_title.prep")
PREP_DATASET_TITLE_PATH_TMP = os.path.join(SCRIPT_PATH, "tmp/dataset_title.prep")

TMP_PATH = os.path.join(SCRIPT_PATH, "tmp/")

PREP_DATASET_INDEX = {
    'original_dataset_size': 802,
    'line_offsets': [0, 243, 585],
    'doc_samples_prefix': None,
    'samples_index': torch.tensor([0, 67, 149, 234, 298, 385]),
    'preprocessed_dataset_size': 480
}

PREP_DATASET_INDEX_TITLE = {
    'original_dataset_size': 802,
    'line_offsets': [0, 243, 585],
    'doc_samples_prefix': [
        torch.tensor([101, 13335, 122, 102], dtype=torch.int32),
        torch.tensor([101, 52294, 123, 102], dtype=torch.int32),
        torch.tensor([101, 52294, 124, 102], dtype=torch.int32)
    ],
    'samples_index': torch.tensor([0, 64, 143, 225, 286, 370]),
    'preprocessed_dataset_size': 462
}

SAMPLES = [(0, torch.tensor([101, 139, 110914, 44376, 97586, 106306, 45007, 13335, 183,
                             124, 172, 11636, 73717, 24204, 10269, 119, 102]), torch.tensor([[7, 7]])),
           (0, torch.tensor([101, 10685, 10138, 12617, 10126, 190, 10157, 52302, 53524, 44703,
                             73683, 10545, 89031, 183, 90086, 95294, 10545, 13796, 20639, 17865,
                             119, 102]), torch.tensor([[10, 12]])),
           (243, torch.tensor([101, 11469, 48185, 189, 32650, 18999, 11169, 11224, 12249, 14231,
                               73718, 83531, 190, 280, 99508, 10138, 10554, 10410, 25964, 29700,
                               10661, 119, 102]), torch.tensor([[10, 10]])),
           (243, torch.tensor([101, 33889, 13426, 10323, 11284, 54609, 53644, 17339, 80521,
                               11246, 75106, 107284, 85955, 184, 119, 102]), torch.tensor([[12, 12]])),
           (243, torch.tensor([101, 41922, 15192, 10330, 108918, 117, 11170, 12132, 103708,
                               190, 10157, 52302, 53524, 31592, 117, 187, 38781, 190,
                               10305, 13520, 78607, 38967, 10138, 119, 102]), torch.tensor([])),
           (585, torch.tensor([101, 148, 35899, 25861, 13061, 10545, 45690, 44286, 64084, 10477,
                               17931, 10545, 153, 119, 11834, 118, 12644, 15119, 10263, 69258,
                               10758, 16874, 42091, 119, 102]), torch.tensor([[1, 5],
                                                                              [6, 6]]))]

SAMPLES_WITH_TITLE = [(0, torch.tensor([101, 13335, 122, 102, 139, 110914, 44376, 97586, 106306, 45007, 13335, 183,
                                        124, 172, 11636, 73717, 24204, 10269, 119, 102]), torch.tensor([[10, 10]])),
                      (0,
                       torch.tensor([101, 13335, 122, 102, 10685, 10138, 12617, 10126, 190, 10157, 52302, 53524, 44703,
                                     73683, 10545, 89031, 183, 90086, 95294, 10545, 13796, 20639, 17865,
                                     119, 102]), torch.tensor([[13, 15]])),
                      (243,
                       torch.tensor([101, 52294, 123, 102, 11469, 48185, 189, 32650, 18999, 11169, 11224, 12249, 14231,
                                     73718, 83531, 190, 280, 99508, 10138, 10554, 10410, 25964, 29700,
                                     10661, 119, 102]), torch.tensor([[13, 13]])),
                      (243, torch.tensor([101, 52294, 123, 102, 33889, 13426, 10323, 11284, 54609, 53644, 17339, 80521,
                                          11246, 75106, 107284, 85955, 184, 119, 102]), torch.tensor([[15, 15]])),
                      (243, torch.tensor([101, 52294, 123, 102, 41922, 15192, 10330, 108918, 117, 11170, 12132, 103708,
                                          190, 10157, 52302, 53524, 31592, 117, 187, 38781, 190,
                                          10305, 13520, 78607, 38967, 10138, 119, 102]), torch.tensor([])),
                      (585,
                       torch.tensor([101, 52294, 124, 102, 148, 35899, 25861, 13061, 10545, 45690, 44286, 64084, 10477,
                                     17931, 10545, 153, 119, 11834, 118, 12644, 15119, 10263, 69258,
                                     10758, 16874, 42091, 119, 102]), torch.tensor([[4, 8],
                                                                                    [9, 9]]))]

DOC_EMBEDDINGS = torch.tensor([
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0]
])


class TestSampleConvertor(unittest.TestCase):
    def test_conversion(self):
        sample = (
            1234,
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([(2, 3)])
        )
        b_rep = SampleConvertor.sample_2_bytes(*sample)
        sample_converted = SampleConvertor.bytes_2_sample(b_rep)

        self.assertEqual(sample[0], sample_converted[0])
        self.assertTrue(isinstance(sample_converted[1], torch.Tensor))
        self.assertTrue(isinstance(sample_converted[2], torch.Tensor))
        self.assertListEqual(sample[1].tolist(), sample_converted[1].tolist())
        self.assertListEqual(sample[2].tolist(), sample_converted[2].tolist())

    def test_conversion_empty_keyphrases(self):
        sample = (
            1234,
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([])
        )
        b_rep = SampleConvertor.sample_2_bytes(*sample)
        sample_converted = SampleConvertor.bytes_2_sample(b_rep)

        self.assertEqual(sample[0], sample_converted[0])
        self.assertTrue(isinstance(sample_converted[1], torch.Tensor))
        self.assertTrue(isinstance(sample_converted[2], torch.Tensor))
        self.assertListEqual(sample[1].tolist(), sample_converted[1].tolist())
        self.assertListEqual(sample[2].tolist(), sample_converted[2].tolist())


class TestPreprocessedDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.index = copy.deepcopy(PREP_DATASET_INDEX)

    def tearDown(self) -> None:
        # remove all in tmp but placeholder
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                os.remove(f)

    def test_init(self):
        prep = PreprocessedDataset(PREP_DATASET_PATH)
        self.assertEqual(PREP_DATASET_PATH, prep.path_to)

        self.assertEqual(self.index['original_dataset_size'], prep._index['original_dataset_size'])
        self.assertEqual(self.index['line_offsets'], prep._index['line_offsets'])
        self.assertEqual(self.index['doc_samples_prefix'], prep._index['doc_samples_prefix'])
        self.assertListEqual(self.index['samples_index'].tolist(), prep._index['samples_index'].tolist())
        self.assertEqual(self.index['preprocessed_dataset_size'], prep._index['preprocessed_dataset_size'])

    def test_init_with_index(self):
        self.index['line_offsets'] = [0, 1, 2]
        prep = PreprocessedDataset(PREP_DATASET_PATH, self.index)
        self.assertEqual(self.index, prep._index)

    def test_init_missing_index(self):
        with self.assertRaises(FileNotFoundError):
            _ = PreprocessedDataset(PREP_NO_INDEX_DATASET_PATH)

    def test_init_diff_prep_dataset(self):
        with self.assertRaises(RuntimeError):
            _ = PreprocessedDataset(PREP_DIFF_PATH)

    def test_with(self):
        dataset = PreprocessedDataset(PREP_DATASET_PATH)
        self.assertIsNone(dataset._dataset_file)
        self.assertIsNone(dataset.opened_in_process_with_id)
        with dataset:
            self.assertEqual(os.getpid(), dataset.opened_in_process_with_id)
            self.assertIsNotNone(dataset._dataset_file)
        self.assertIsNone(dataset._dataset_file)
        self.assertIsNone(dataset.opened_in_process_with_id)

    def test_open_close(self):
        dataset = PreprocessedDataset(PREP_DATASET_PATH)
        self.assertIsNone(dataset._dataset_file)
        self.assertIsNone(dataset.opened_in_process_with_id)
        dataset.open()
        self.assertIsNotNone(dataset._dataset_file)
        self.assertEqual(os.getpid(), dataset.opened_in_process_with_id)
        dataset.close()
        self.assertIsNone(dataset._dataset_file)
        self.assertIsNone(dataset.opened_in_process_with_id)

    def test_from_dataset(self):
        dataset = PreprocessedDataset.from_dataset(DATASET_PATH, PREP_DATASET_PATH_TMP, use_title=False,
                                                   splitter=CzechLibrarySampleSplitter(
                                                       AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                                                   ))
        self.assertTrue(Path(PREP_DATASET_PATH_TMP).exists())
        self.assertTrue(Path(PREP_DATASET_PATH_TMP + ".index").exists())

        self.assertEqual(self.index['original_dataset_size'], dataset._index['original_dataset_size'])
        self.assertEqual(self.index['line_offsets'], dataset._index['line_offsets'])
        self.assertEqual(self.index['doc_samples_prefix'], dataset._index['doc_samples_prefix'])
        self.assertListEqual(self.index['samples_index'].tolist(), dataset._index['samples_index'].tolist())
        self.assertEqual(self.index['preprocessed_dataset_size'], dataset._index['preprocessed_dataset_size'])

        # check on content lvl
        with open(PREP_DATASET_PATH, "rb") as gt_f, open(PREP_DATASET_PATH_TMP, "rb") as res_f:
            gt_prep_data = gt_f.read()
            res_prep_data = res_f.read()

            self.assertEqual(gt_prep_data, res_prep_data)


class TestPreprocessedDatasetWithTitle(unittest.TestCase):
    def setUp(self) -> None:
        self.index = copy.deepcopy(PREP_DATASET_INDEX_TITLE)

    def tearDown(self) -> None:
        # remove all in tmp but placeholder
        for f in Path(TMP_PATH).glob('*'):
            if not str(f).endswith("placeholder"):
                os.remove(f)

    def test_init(self):
        prep = PreprocessedDataset(PREP_DATASET_TITLE_PATH)
        self.assertEqual(PREP_DATASET_TITLE_PATH, prep.path_to)

        self.assertEqual(self.index['original_dataset_size'], prep._index['original_dataset_size'])
        self.assertEqual(self.index['line_offsets'], prep._index['line_offsets'])
        if self.index['doc_samples_prefix'] is None:
            self.assertIsNone(prep._index['doc_samples_prefix'])
        else:
            self.assertListEqual([x.tolist() for x in self.index['doc_samples_prefix']],
                                 [x.tolist() for x in prep._index['doc_samples_prefix']])
        self.assertListEqual(self.index['samples_index'].tolist(), prep._index['samples_index'].tolist())
        self.assertEqual(self.index['preprocessed_dataset_size'], prep._index['preprocessed_dataset_size'])

    def test_init_with_index(self):
        self.index['line_offsets'] = [0, 1, 2]
        prep = PreprocessedDataset(PREP_DATASET_TITLE_PATH, self.index)
        self.assertEqual(self.index, prep._index)

    def test_from_dataset(self):
        dataset = PreprocessedDataset.from_dataset(DATASET_TITLE_PATH, PREP_DATASET_TITLE_PATH_TMP, use_title=True,
                                                   splitter=CzechLibrarySampleSplitter(
                                                       AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                                                   ))
        self.assertTrue(Path(PREP_DATASET_TITLE_PATH_TMP).exists())
        self.assertTrue(Path(PREP_DATASET_TITLE_PATH_TMP + ".index").exists())

        self.assertEqual(self.index['original_dataset_size'], dataset._index['original_dataset_size'])
        self.assertEqual(self.index['line_offsets'], dataset._index['line_offsets'])
        if self.index['doc_samples_prefix'] is None:
            self.assertIsNone(dataset._index['doc_samples_prefix'])
        else:
            self.assertListEqual([x.tolist() for x in self.index['doc_samples_prefix']],
                                 [x.tolist() for x in dataset._index['doc_samples_prefix']])
        self.assertListEqual(self.index['samples_index'].tolist(), dataset._index['samples_index'].tolist())
        self.assertEqual(self.index['preprocessed_dataset_size'], dataset._index['preprocessed_dataset_size'])

        # check on content lvl
        with open(PREP_DATASET_TITLE_PATH, "rb") as gt_f, open(PREP_DATASET_TITLE_PATH_TMP, "rb") as res_f:
            gt_prep_data = gt_f.read()
            res_prep_data = res_f.read()

            self.assertEqual(gt_prep_data, res_prep_data)


class TestsOnAlreadyOpenedPreprocessedDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.index = copy.deepcopy(PREP_DATASET_INDEX)
        self.dataset = PreprocessedDataset(PREP_DATASET_PATH, self.index)
        self.dataset.open()

        self.samples = copy.deepcopy(SAMPLES)

    def tearDown(self) -> None:
        self.dataset.close()

    def test_path_to(self):
        self.assertEqual(PREP_DATASET_PATH, self.dataset.path_to)

    def test_len(self):
        self.assertEqual(len(self.samples), len(self.dataset))

    def test_get_item(self):
        for gt_sample, res_sample in zip(self.samples, self.dataset):
            self.assertEqual(gt_sample[0], res_sample[0])  # line offset
            self.assertListEqual(gt_sample[1].tolist(), res_sample[1].tolist())  # tokens
            self.assertListEqual(gt_sample[2].tolist(), res_sample[2].tolist())  # spans

    def test_get_item_with_title(self):
        for gt_sample, res_sample in zip(self.samples, self.dataset):
            self.assertEqual(gt_sample[0], res_sample[0])  # line offset
            self.assertListEqual(gt_sample[1].tolist(), res_sample[1].tolist())  # tokens
            self.assertListEqual(gt_sample[2].tolist(), res_sample[2].tolist())  # spans

    def test_check_match_with(self):
        self.dataset.check_match_with(DATASET_PATH)

        with self.assertRaises(RuntimeError):
            self.dataset.check_match_with(TEST_PATH)


class TestsOnAlreadyOpenedPreprocessedDatasetWithTitle(unittest.TestCase):
    def setUp(self) -> None:
        self.index = copy.deepcopy(PREP_DATASET_INDEX_TITLE)
        self.dataset = PreprocessedDataset(PREP_DATASET_TITLE_PATH, self.index)
        self.dataset.open()

        self.samples = copy.deepcopy(SAMPLES_WITH_TITLE)

    def tearDown(self) -> None:
        self.dataset.close()

    def test_len(self):
        self.assertEqual(len(self.samples), len(self.dataset))

    def test_get_item(self):
        for gt_sample, res_sample in zip(self.samples, self.dataset):
            self.assertEqual(gt_sample[0], res_sample[0])  # line offset
            self.assertListEqual(gt_sample[1].tolist(), res_sample[1].tolist())  # tokens
            self.assertListEqual(gt_sample[2].tolist(), res_sample[2].tolist())  # spans

    def test_get_item_with_title(self):
        for gt_sample, res_sample in zip(self.samples, self.dataset):
            self.assertEqual(gt_sample[0], res_sample[0])  # line offset
            self.assertListEqual(gt_sample[1].tolist(), res_sample[1].tolist())  # tokens
            self.assertListEqual(gt_sample[2].tolist(), res_sample[2].tolist())  # spans

    def test_check_match_with(self):
        self.dataset.check_match_with(DATASET_TITLE_PATH)

        with self.assertRaises(RuntimeError):
            self.dataset.check_match_with(TEST_PATH)


if __name__ == '__main__':
    unittest.main()
