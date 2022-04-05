# -*- coding: UTF-8 -*-
""""
Created on 11.04.21

:author:     Martin Dočekal
"""
import datetime
import os
import shutil
import socket
import unittest
from io import StringIO
from pathlib import Path

from typing import List

from qbek.entities import DocumentResults, Document
from qbek.results import ResultsSaving


class MockModel:
    def __init__(self, current_epoch: int, global_step: int):
        self.current_epoch = current_epoch
        self.global_step = global_step


class MockCzechLibrary:

    def line(self, line_offset):
        return f'{{"uuid": {line_offset}, ' \
               f'"query": "Title {line_offset}", ' \
               f'"keyphrases": ["kp{line_offset}", "kp"], ' \
               f'"contexts": ["kp{line_offset}\\t0 3"]}}'


class MockCzechLibraryWithContent:

    def __init__(self, lines: List[str]):
        self.lines = lines

    def line(self, line_offset):
        return self.lines[line_offset]


class TestResultsSaving(unittest.TestCase):
    path_to_this_script_file = os.path.dirname(os.path.realpath(__file__))
    tmp_path = os.path.join(path_to_this_script_file, "tmp/")

    def setUp(self) -> None:
        self.results_saving = ResultsSaving("exp_name", MockModel(10, 100), self.tmp_path)

    def tearDown(self) -> None:
        # remove all in tmp but placeholder
        for f in Path(self.tmp_path).glob('*'):
            if not str(f).endswith("placeholder"):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

    def test_call(self):
        class MockSaveResultsForDataset:
            def __init__(self):
                self.outputs = None
                self.dataset = None
                self.f = None

            def __call__(self, outputs, dataset, f):
                self.outputs = outputs
                self.dataset = dataset
                self.f = f

        mock = MockSaveResultsForDataset()
        self.results_saving.save_results_for_dataset = mock

        outputs = [[("kp", 5.0, 0)]]

        path = self.results_saving(outputs, MockCzechLibrary(), prefix="p")

        self.assertListEqual(mock.outputs, outputs)
        self.assertTrue(isinstance(mock.dataset, MockCzechLibrary))
        self.assertIsNotNone(mock.f)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+"_"+socket.gethostname()
        self.assertEqual(path, os.path.join(self.tmp_path, "exp_name", f"p_e10_s100_{stamp}.tsv"))

    def test_save_results_for_dataset(self):
        outputs = [
            DocumentResults(documents_line_offset=0,
                            spans=["kp1", "kp2", "kp3", "kp4"],
                            scores=[5.0, 11.0, 15.0, 10.0]),
            DocumentResults(documents_line_offset=10,
                            spans=[],
                            scores=[])
        ]
        output_f = StringIO()

        self.results_saving.save_results_for_dataset(outputs, MockCzechLibrary(), output_f)

        self.assertEqual(output_f.getvalue(),
                         "uuid\tquery\tannotated\textractive ground truth\tpredicted\r\n"
                         "0\tTitle 0\tkp█kp0\tkp0\tkp3█kp2█kp4█kp1\r\n"
                         "10\tTitle 10\tkp█kp10\t\t\r\n")


if __name__ == '__main__':
    unittest.main()
