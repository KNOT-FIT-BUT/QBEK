# -*- coding: UTF-8 -*-
""""
Created on 05.04.2021
Module containing metrics for keyphrases extraction/generation evaluation.

:author:     Martin Dočekal
"""
from io import StringIO

from typing import Tuple, Optional, TextIO, Iterable, Set, Callable, Dict, Sequence

from qbek.lemmatizer import Lemmatizer
from qbek.tokenizer import Tokenizer


class Evaluation:
    """
    Functor for keyphrases evaluation.
    """

    def __init__(self, name: str, transformation: Callable[[Set[str]], Set[str]]):
        """
        inits evaluation

        :param name: name of evaluation kind
        :param transformation: transformation that should be used on keyphrases
            e.g. some kind of normalization
        """
        self.name = name
        self.transformation = transformation
        self.correct_cnt = 0
        self.extracted_cnt = 0
        self.ground_truth_cnt = 0

    def __call__(self, pred: Set[str], gt: Set[str]):
        """
        Updates metrics.

        :param pred: predicted values
        :param gt: ground truth values.
        """

        pred = self.transformation(pred)
        gt = self.transformation(gt)

        self.extracted_cnt += len(pred)
        self.ground_truth_cnt += len(gt)
        self.correct_cnt += len(pred & gt)

    def precision_recall_f1(self) -> Tuple[float, float, float]:
        """
        Returns current precision recall f1.

        :return: precision, recall, f1
        precision and recall are calculated according to following table

        c   e   g       p   r
        0   0   0       1   1
        0   0   >0      0   0
        0   >0  0       0   0
        0   >0  >0      0   0
        >0  0   0       not exists
        >0  0   >0      not exists
        >0  >0  0       not exists
        >0  >0  >0      c/e c/g

        where
            c is number of correctly predicted keyphrases
            e is number of predicted keyphrases
            g is number of ground truth keyphrases
        """
        if self.correct_cnt == 0:
            if self.extracted_cnt == self.ground_truth_cnt == 0:
                return 1.0, 1.0, 1.0

            return 0, 0, 0

        precision = (self.correct_cnt / self.extracted_cnt) if self.extracted_cnt > 0 else 0
        recall = (self.correct_cnt / self.ground_truth_cnt) if self.ground_truth_cnt > 0 else 0

        return precision, recall, 2 * precision * recall / (precision + recall)


class ExtractionGenerationMetric:
    """
    Abstract base class for extraction/generation metrics on document level.
    """
    EVALUATION_TYPE_NAMES = ["exact", "lower case", "lemma", "lower case lemma"]

    def __init__(self, tokenizer: Tokenizer, lemmatizer: Lemmatizer):
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer
        self.evaluation_types = self.EVALUATION_TYPE_NAMES[:]
        self.transformations_for_types = {
            "exact": lambda x: self.tokenize(x),
            "lower case": lambda x: self.lower(self.tokenize(x)),
            "lemma": lambda x: self.lemmatize(self.tokenize(x)),
            "lower case lemma": lambda x: self.lower(self.lemmatize(self.tokenize(x)))
        }

    def eval(self, spans: Iterable[Iterable[str]], gt_spans: Iterable[Iterable[str]]) \
            -> Dict[str, Tuple[float, float, float]]:
        """
        Evaluation of results.

        :param spans: Predicted spans in form of strings for each dataset sample.
        :param gt_spans: Ground truth spans in form of strings for each dataset sample.
        :return: precision, recall and f1 for each of evaluation types
        To see how precision and recall are calculated see :func:`~Evaluation.precision_recall_f1`.
        """

        evaluators = [
            Evaluation(name, self.transformations_for_types[name]) for name in self.evaluation_types
        ]
        for doc_pred, doc_gt in zip(spans, gt_spans):
            for e in evaluators:
                e(set(doc_pred), set(doc_gt))   # update evaluators

        return {e.name: e.precision_recall_f1() for e in evaluators}

    def tokenize(self, keyphrases: Set[str]) -> Set[Tuple[str, ...]]:
        """
        Tokenization of keyphrases.

        :param keyphrases: keyphrases you want to convert
        :return: tokens of given kyphrases
        """

        return {self.tokenizer(k) for k in keyphrases}

    def lemmatize(self, keyphrases: Set[Sequence[str]]) -> Set[str]:
        """
        Lemmatization of keyphrases.

        :param keyphrases: keyphrases you want to convert
        :return: lemmatized forms of given kyphrases
        """

        return {self.lemmatizer.lemmatize(k) for k in keyphrases}

    @staticmethod
    def lower(keyphrases: Set[Sequence[str]]) -> Set[Tuple[str, ...]]:
        """
        Converts keyphrases to lower case.

        :param keyphrases: keyphrases you want to convert
        :return: lower case forms of kyphrases
        """

        return {tuple(t.lower() for t in k) for k in keyphrases}

    @classmethod
    def print_eval_res(cls, res: Dict[str, Tuple[float, float, float]], out: Optional[TextIO] = None) -> Optional[str]:
        """
        Prints metrics from evaluations.


        :param res: Results of evaluation.
            precision, recall and f1 for each of evaluation types
        :param out: Where you want to print
            If none returns string
        :return: Of out is none returns string that will be printed out
        """
        return_str = False
        if out is None:
            return_str = True
            out = StringIO()

        for type_name in cls.EVALUATION_TYPE_NAMES:
            print(f"{type_name}:", file=out)
            print(f"\tprecision:\t{res[type_name][0]}", file=out)
            print(f"\trecall:\t{res[type_name][1]}", file=out)
            print(f"\tf1:\t{res[type_name][2]}", file=out)

        if return_str:
            return out.getvalue()


