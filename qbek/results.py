# -*- coding: UTF-8 -*-
""""
Created on 06.04.21
Module for results manipulation.

:author:     Martin Dočekal
"""
import csv
import datetime
import os
import socket
from pathlib import Path
from typing import List, TextIO, Iterable

from qbek.datasets.czech_library import CzechLibrary
from qbek.entities import Document, DocumentResults
from qbek.models.model import Model


class ResultsSaving:
    """
    Class for saving results.
    """

    KEYPHRASES_SEPARATOR = "█"

    def __init__(self, experiments_name: str, module: Model, results: str):
        """
        Initialization of results saving.

        :param experiments_name: experiment's name that will be part of results file name
        :param module: This model will be used to get current epoch and global step.
        :param results: Path to directory where results should be saved.
            A sub-folder with experiment's name will be created.
        """

        self._experiments_name = experiments_name
        self._module = module
        self._results = os.path.join(results, experiments_name)
        Path(self._results).mkdir(parents=True, exist_ok=True)

    def __call__(self, outputs: Iterable[DocumentResults], dataset: CzechLibrary,
                 prefix: str = "") -> str:

        """
        saves outputs to results

        :param outputs: outputs from model
        :param dataset: outputs are for this dataset
        :param prefix: file name prefix
        :return: Path to results
        """

        file_name = f"{prefix}{self._file_name_postfix()}"
        path = os.path.join(self._results, file_name)
        with open(path, "w") as f:
            self.save_results_for_dataset(outputs, dataset, f)
        return path

    @classmethod
    def save_results_for_dataset(cls, outputs: Iterable[DocumentResults], dataset: CzechLibrary,
                                 f: TextIO):
        """
        Saves results for given dataset to the output file.

        :param outputs: outputs from model.
        :param dataset: For that dataset the outputs were created
        :param f: file where the results should be saved
        """

        print("uuid\tquery\tannotated\textractive ground truth\tpredicted", file=f, end="\r\n")
        for doc_results in outputs:
            document = Document.from_json(dataset.line(doc_results.documents_line_offset))
            cls._print_sample(document, doc_results, f)

    @classmethod
    def _print_sample(cls, document: Document, results: DocumentResults, f: TextIO):
        """
        Prints results for dataset samples to the file.

        :param document: The document for which the result are.
        :param results: results for the document
        :param f: file where the results should be saved
        """

        writer = csv.writer(f, delimiter='\t')

        annotated = sorted(set(document.annotated_keyphrases))
        spans = list(s[1] for s in sorted(enumerate(results.spans), key=lambda x: results.scores[x[0]], reverse=True))
        ext_kp = sorted(set(document.annotated_ext_kp))

        writer.writerow([
            document.uuid, document.title_statement, cls.KEYPHRASES_SEPARATOR.join(annotated),
            cls.KEYPHRASES_SEPARATOR.join(ext_kp), cls.KEYPHRASES_SEPARATOR.join(spans)
        ])

    def _file_name_postfix(self):
        return f"_e{self._module.current_epoch}_s{self._module.global_step}_{self._stamp()}.tsv"

    @staticmethod
    def _stamp() -> str:
        """
        current stamp
        """

        return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+"_"+socket.gethostname()
