# -*- coding: UTF-8 -*-
""""
Created on 05.04.2021
Module containing metrics for keyphrases identification evaluation.

:author:     Martin Dočekal
"""
from abc import abstractmethod, ABC
from io import StringIO
from typing import Iterable, Tuple, Dict, Any, Optional, List, TextIO

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from windpyutils.structures.span_set import SpanSetExactEqRelation, SpanSetPartOfEqRelation, SpanSetIncludesEqRelation, \
    SpanSetOverlapsEqRelation, SpanSet, SpanSetEqRelation


class IdentificationMetric(ABC):
    """
    Abstract base class for identification metrics.
    """

    @abstractmethod
    def eval(self, spans: Iterable[Iterable[Tuple[int, int]]], gt_spans: Iterable[Iterable[Tuple[int, int]]]):
        """
        Evaluation of results.

        :param spans: Predicted spans in form of tuples (start offset, end offset) for each sample.
        :param gt_spans: Ground truth spans in form of tuples (start offset, end offset) for each sample.
        :return: evaluation
        """
        pass


class MultiLabelIdentificationMetrics(IdentificationMetric):
    """
    Metrics based on the to multi-label example based scheme presented in:
        A Review on Multi-Label Learning Algorithms
        url: https://ieeexplore.ieee.org/document/6471714

    Provided metrics are:
                "exactMatchWithAny":
                    predicted span is the same as at least one known span
                    Example:
                        Known span: 10 20                   10 ------------- 20
                        Predicted span: 10 20               10 ------------- 20
                "partOfAny":
                    predicted span is in any known span
                    Example:
                        Known span: 10 20                   10 ------------- 20
                        Predicted span: 15 18                     15 ---- 18
                "includesAny":
                    any known span is in predicted span
                    Example:
                        Known span: 15 18                          15 ---- 18
                        Predicted span: 10 20               10 ------------- 20
                "sharedPartOfSpan":
                    predicted span has non empty intersection with any of known spans
                    Example:
                        Known span: 15 18                   15 ---- 18
                        Predicted span: 17 20                     17 ------------- 20
                "noMatchAtAll":
                    Number of samples that haves no match with any of the known spans.

    :ivar metrics: Metrics that should be calculated.
        noMatchAtAll will be evaluated no matter what
    :vartype metrics: List[Tuple[str, SpanSetEqRelation]]
    """

    def __init__(self, metrics: Optional[List[Tuple[str, SpanSetEqRelation]]] = None):
        """

        :param metrics: Default is None which means that all metrics are counted.
            You can choose only certain metrics that should be evaluated.
            noMatchAtAll will be evaluated no matter what
        """

        if metrics is None:
            self.metrics = [("exactMatchWithAny", SpanSetExactEqRelation()),
                            ("partOfAny", SpanSetPartOfEqRelation()),
                            ("includesAny", SpanSetIncludesEqRelation()),
                            ("sharedPartOfSpan", SpanSetOverlapsEqRelation())]
        else:
            self.metrics = metrics

    def eval(self, spans: Iterable[Iterable[Tuple[int, int]]], gt_spans: Iterable[Iterable[Tuple[int, int]]]) -> Dict:
        """
        Evaluation of results.

        :param spans: Predicted spans in form of tuples (start offset, end offset) for each sample.
            Spans are right opened intervals.
        :param gt_spans: Ground truth spans in form of tuples (start offset, end offset) for each sample.
            Spans are right opened intervals.
        :return: Dictionary of metrics. Each metric (but the noMatchAtAll) contains dict
            with accuracy, precision, recall and f1.

        """

        # we will calc metrics according to multi-label example based scheme presented in:
        #   A Review on Multi-Label Learning Algorithms
        #       url: https://ieeexplore.ieee.org/document/6471714

        res = {
            "noMatchAtAll": 0,
            "empty": None,
        }
        for m, _ in self.metrics:
            res[m] = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        number_of_results = 0

        # let's convert right opened intervals, to close intervals that SpanSet expects
        spans = [[(s, e-1) for s, e in sample] for sample in spans]
        gt_spans = [[(s, e - 1) for s, e in sample] for sample in gt_spans]

        empty_pred = []
        empty_gt = []
        for i, (samples_pred_spans, samples_ground_truth_spans) in enumerate(zip(spans, gt_spans)):
            number_of_results += 1
            # We will be using special set SpanSet, which is set that allows to change common behaviour of in operator.
            # We will use this feature to calc all metric that we need. Because this concept isn't to much straightforward
            # let me explain it on the example.
            #
            # Imagine that you need to calc partOfAny accuracy, which means that you are calculating accuracy, where not
            # just exact matches are ok, but also spans that fits into any of known (ground truth) spans.
            #   Accuracy for that type of match is calculated as (according to article mentioned earlier):
            #           1/n (Σ (|Y_i ∩ h(x_i)| / |Y_i ∪ h(x_i)|))
            #   Where Y_i is set of ground truth spans and h(x_i) is set of predicted spans for sample i. The n is
            #   number of all samples and summation goes truth every sample.
            # So ok, there is set intersection and union of two sets.
            # The definition of intersection for two sets is: A ∩ B = { x ∣ x∈A ∧ x∈B }
            # where ∈ is custom operator for a given set, that we are overloading.
            # If B is set of predicted spans (B=h(x_i)) and A is set of ground truth spans (A=Y_i), we can get the set
            # of all predicted spans that suits the condition (partOfAny) by overloading the ∈ for A=Y_i in a way that
            # the operator returns True for all spans that suits the condition. For example if we predict span (2,3)
            # and the A set consists of {(1,4), (5,8)}, the ∈ operator returns True because A set contains (1,4).
            # We are leaving the operator ∈ for set B to return true for exact matches only (default behaviour),
            # because we want to have in results spans that we really predicted.
            # The behaviour of ∈ could be overloaded by providing different relation of equality (SpanSetEqRelation).
            # The union (Y_i ∪ h(x_i)) is bigger problem, because we can not just overload the in operator like for the intersection as shows
            # that example:
            #   Y_i = {(0,1), (0,3)}
            #   h(x_i) = {(1,3), (2,2)}
            #   Y_i ∩ h(x_i) = { x ∣ x∈Y_i ∧ x∈h(x_i) } = {(0,1), (0,3)} # x∈Y_i uses special in operator
            #   Y_i ∪ h(x_i) = { x ∣ x∈Y_i ∨ x∈h(x_i) } = {(0,1), (0,3), (1,3), (2,2)} # x∈Y_i uses special in operator
            #   which means that accuracy is:
            #       1/n (Σ (|Y_i ∩ h(x_i)| / |Y_i ∪ h(x_i)|)) = 0.5
            #   which is od because the precision and recall (according to formulas from https://ieeexplore.ieee.org/document/6471714 ):
            #       precision:
            #           1/n (Σ (|Y_i ∩ h(x_i)| / |h(x_i)|)) = 1
            #       recall:
            #           1/n (Σ (|Y_i ∩ h(x_i)| / |Y_i|)) = 1
            #
            #   So instead we calc the |Y_i ∪ h(x_i)| with usage of inclusion-exclusion principle as:
            #       |Y_i| + |h(x_i)| - |Y_i ∩ h(x_i)|
            #   It can be read as sum of number of all labels and number of wrongly predicted labels (|h(x_i)| - |Y_i ∩ h(x_i)|)

            predicted_spans_set = SpanSet(samples_pred_spans, force_no_dup_check=True)
            gt_spans_set = SpanSet(samples_ground_truth_spans, force_no_dup_check=True)

            empty_pred.append(len(predicted_spans_set) == 0)
            empty_gt.append(len(gt_spans_set) == 0)

            # Ok we have all we need, there is just the right time to calc all metrics.
            no_match = True
            for match_type, eq_rel_set in self.metrics:
                gt_spans_set.eq_relation = eq_rel_set

                common_spans = predicted_spans_set & gt_spans_set

                if no_match and (len(common_spans) > 0 or len(gt_spans_set) == len(predicted_spans_set) == 0):
                    # we have at least one common span or there should be no spans predicted and we predicted no spans
                    no_match = False

                if len(gt_spans_set) == len(predicted_spans_set) == 0:
                    res[match_type]["accuracy"] += 1
                else:
                    # the inclusion-exclusion principle
                    res[match_type]["accuracy"] += len(common_spans) / (
                            len(gt_spans_set) + len(predicted_spans_set) - len(common_spans))

                if len(predicted_spans_set) == 0:
                    if len(gt_spans_set) == 0:
                        # Predicted spans set is empty and the set of known spans is empty too, that's good.
                        res[match_type]["precision"] += 1
                    else:
                        # precision determines fraction of correctly predicted labels from all predicted labels
                        # because there are no predicted labels, but some should be predicted we are setting it to zero
                        res[match_type]["precision"] += 0
                else:
                    res[match_type]["precision"] += len(common_spans) / len(predicted_spans_set)

                if len(gt_spans_set) == 0:
                    if len(predicted_spans_set) == 0:
                        # Predicted spans set is empty and the set of known spans is empty too, that's good.
                        res[match_type]["recall"] += 1
                    else:
                        # recall determines fraction of correctly predicted labels from all correct known labels
                        # because there are no predicted labels, but some should be predicted we are setting it to zero
                        res[match_type]["recall"] += 0
                else:
                    # Recall can be here grater than one because some matching strategies could add to common_spans
                    # some additional spans and then common_spans set can be grater than gt_spans_set
                    res[match_type]["recall"] += len(common_spans) / len(gt_spans_set)

            if no_match:
                # predicted keywords doesn't match with any ground truth keywords no mater the match strategy
                res["noMatchAtAll"] += 1

        # we have just the sums from the metrics equations and no f1
        # so let's finished it
        for k, m in res.items():
            if k != "noMatchAtAll" and k != "empty":
                m["accuracy"] /= number_of_results
                m["precision"] /= number_of_results
                m["recall"] /= number_of_results
                m["f1"] = 0 if m["precision"] == 0 and m["recall"] == 0 else 2.0 * m["precision"] * m["recall"] / (
                        m["precision"] + m["recall"])

        res["empty"] = (accuracy_score(empty_gt, empty_pred), ) + \
                       precision_recall_fscore_support(empty_gt, empty_pred, average='binary')

        return res

    @staticmethod
    def print_eval_res(res: Dict[str, Any], out: Optional[TextIO] = None) -> Optional[str]:
        """
        Prints metrics from evaluations.


        :param res: Results of evaluation.
        :param out: Where you want to print
            If none returns string
        :return: Of out is none returns string that will be printed out
        """
        return_str = False
        if out is None:
            return_str = True
            out = StringIO()

        for k, desc in [("exactMatchWithAny", "exact match with any"), ("partOfAny", "is part of any"),
                        ("includesAny", "includes any"), ("sharedPartOfSpan", "shared part of span with any")]:
            if k in res:
                print("\t" + desc, file=out)
                print("\t\taccuracy:\t{}".format(res[k]["accuracy"]), file=out)
                print("\t\tprecision:\t{}".format(res[k]["precision"]), file=out)
                print("\t\trecall:\t{}".format(res[k]["recall"]), file=out)
                print("\t\tf1:\t{}".format(res[k]["f1"]), file=out)

        print("\tno match at all", file=out)
        print("\t\t{}".format(res["noMatchAtAll"]), file=out)

        print("\tempty", file=out)
        print("\t\taccuracy:\t{}".format(res["empty"][0]), file=out)
        print("\t\tprecision:\t{}".format(res["empty"][1]), file=out)
        print("\t\trecall:\t{}".format(res["empty"][2]), file=out)
        print("\t\tf1:\t{}".format(res["empty"][3]), file=out)

        if return_str:
            return out.getvalue()

'''
TODO: Remove
class SemEvalIdentificationMetric(IdentificationMetric):
    """
    Returns metrics as are is calculated in SemEval2017 task 10.
    """

    def eval(self, spans: Iterable[Iterable[Tuple[int, int]]], gt_spans: Iterable[Iterable[Tuple[int, int]]]):
        """
        Evaluation of results.

        :param spans: Predicted spans in form of tuples (start offset, end offset) for each sample.
        :param gt_spans: Ground truth spans in form of tuples (start offset, end offset) for each sample.
        :return: Dictionary of metrics. Each metric (but the noMatchAtAll) contains dict
            with accuracy, precision, recall and f1.
            Provided metrics are:
                "exactMatchWithAny":
                    predicted span is the same as at least one known span
                    Example:
                        Known span: 10 20                   10 ------------- 20
                        Predicted span: 10 20               10 ------------- 20
                "partOfAny":
                    predicted span is in any known span
                    Example:
                        Known span: 10 20                   10 ------------- 20
                        Predicted span: 15 18                     15 ---- 18
                "includesAny":
                    any known span is in predicted span
                    Example:
                        Known span: 15 18                          15 ---- 18
                        Predicted span: 10 20               10 ------------- 20
                "sharedPartOfSpan":
                    predicted span has non empty intersection with any of known spans
                    Example:
                        Known span: 15 18                   15 ---- 18
                        Predicted span: 17 20                     17 ------------- 20
                "noMatchAtAll":
                    Number of samples that haves no match with any of the known spans.
        """

        metrics = [("exactMatchWithAny", SpanSetExactEqRelation()),
                   ("partOfAny", SpanSetPartOfEqRelation()),
                   ("includesAny", SpanSetIncludesEqRelation()),
                   ("sharedPartOfSpan", SpanSetOverlapsEqRelation())]
        res = {
            "noMatchAtAll": 0
        }

        res_all_gold = {}
        res_all_pred = {}
        for m, _ in metrics:
            res_all_gold[m] = []
            res_all_pred[m] = []

        number_of_results = 0
        for i, (samples_pred_spans, samples_ground_truth_spans) in enumerate(zip(spans, gt_spans)):
            number_of_results += 1

            predicted_spans_set = SpanSet(samples_pred_spans, force_no_dup_check=True)
            gt_spans_set = SpanSet(samples_ground_truth_spans, force_no_dup_check=True)

            all_spans = predicted_spans_set | gt_spans_set

            # Ok we have all we need, there is just the right time to calc all metrics.
            no_match = True
            for match_type, eq_rel_set in metrics:
                gt_spans_set.eqRelation = eq_rel_set
                for span in all_spans:
                    span_in_gt = False
                    if span in gt_spans_set:
                        # true positives
                        res_all_gold[match_type].append(1)

                        span_in_gt = True
                    else:
                        # false positives
                        res_all_gold[match_type].append(0)

                    if span in predicted_spans_set:
                        if span_in_gt:
                            no_match = False
                        res_all_pred[match_type].append(1)
                    else:
                        # false negatives
                        res_all_pred[match_type].append(0)

            if no_match:
                # predicted keywords doesn't match with any ground truth keywords no mater the match strategy
                res["noMatchAtAll"] += 1

        for m, _ in metrics:
            p, r, f1, _ = precision_recall_fscore_support(res_all_gold[m], res_all_pred[m], labels=[0, 1], average=None)
            res[m] = {
                "accuracy": accuracy_score(res_all_gold[m], res_all_pred[m]),
                "precision": p[1],
                "recall": r[1],
                "f1": f1[1],
            }

        return res
'''
