# -*- coding: UTF-8 -*-
""""
Created on 02.11.21

:author:     Martin Dočekal
"""
import itertools
import math
import threading
from collections import defaultdict
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool
from typing import Optional, List, Tuple, Set, Union

import hyperopt
import numpy as np
import torch
from numba import njit
from tqdm import tqdm
from windpyutils.visual.text import print_buckets_histogram

from qbek.batch import Batch
from qbek.hyper_tune import FilterWrapperForHyperTune
from qbek.lemmatizer import Lemmatizer
from qbek.metrics.extraction_generation import ExtractionGenerationMetric
from qbek.models.model import Model
from qbek.predictions_filters import CompoundFilter, SpanSizeFilter, ThresholdFilter
from qbek.tokenizer import AlphaTokenizer
from qbek.utils.top_keeper import TopKeeper


class AutoFilter:
    """
    Auto filter finder for a model.

    :ivar n_best: If none the n best filter will be found automatically, else the given value will be used.
    :vartype n_best: Optional[int]
    :ivar max_span_size: If none the max span size filter will be found automatically,
        else the given value will be used.
    :vartype max_span_size: Optional[int]
    :ivar score_threshold: If none the score threshold filter will be found automatically,
        else the given value will be used.
    :vartype score_threshold: Optional[float]
    :ivar metric: Filters will be evaluated with this metric.
    :vartype metric: IdentificationMetric
    """

    THRESHOLDS_SPACE = 25
    """
    number of thresholds that will be tried when auto searching the best threshold
    
    Example:
     min_score: 0
     max_score: 1
     THRESHOLDS_SPACE: 4
     tries: 0 0.25, 0.25, 0.75
    """

    def __init__(self, lemmatizer: Lemmatizer, n_best: Optional[int] = None, max_span_size: Optional[int] = None,
                 score_threshold: Optional[float] = None, trials: int = 100):
        """
        Initialization of auto filter finder.

        :param lemmatizer: Lemmatizer that should be used for normalization during evaluation.
        :param n_best: If none the n best filter will be found automatically, else the given value will be used.
        :param max_span_size: If none the max span size filter will be found automatically,
            else the given value will be used.
        :param score_threshold: If none the score threshold filter will be found automatically,
            else the given value will be used.
        :param trials: number of trials when searching the best filters combination.
        """

        self.n_best = n_best
        self.max_span_size = max_span_size
        self.score_threshold = score_threshold
        self.metric = ExtractionGenerationMetric(AlphaTokenizer(), lemmatizer)
        self.metric.evaluation_types = ["lower case lemma"]
        self._ground_truths = None
        self._predictions = None
        self._batches = None
        self._trials = trials

        self._use_devices = [
                torch.device("cuda", i) for i in range(torch.cuda.device_count())
            ]

    def all_params_known(self) -> bool:
        """
        Tels whether all needed params are known.
        The __call__ needs no data (you can pass empty list).

        :return: True if all params are known.
        """
        return all(x is not None for x in [self.n_best, self.max_span_size, self.score_threshold])

    def __call__(self, predictions: List[List[Tuple[torch.Tensor, Batch]]], ground_truths: List[Set[str]]) \
            -> Tuple[int, float, int, Optional[float]]:
        """
        Finds best settings of filters.

        :param predictions: Predictions for given dataset in form of list of list of tuples
            there is for each document list containing: (prediction in form of span universe matrix, batch).
        :param ground_truths: Extractive ground truths for each documents.
        :return: best filtration combination  (max_span_size, score_threshold, n_best, F1 (for auto else None))
        """
        if self.all_params_known():
            return self.max_span_size, self.score_threshold, self.n_best, None

        span_lengths = defaultdict(int)
        ext_keyphrases_per_doc = defaultdict(int)

        min_score = math.inf
        max_score = -math.inf

        # both are pooled on document level, so element [0] will contain all  ground truths / prediction for
        # document 0
        self._ground_truths = ground_truths
        self._predictions = [[]]
        self._batches = [[]]

        for doc_i, document_res in enumerate(predictions):
            for batch_span_universe_matrix_predictions, batch in document_res:
                batch: Batch
                if isinstance(batch_span_universe_matrix_predictions, np.ndarray):
                    # for some reason torch lightning was converting torch tenor to numpy array
                    batch_span_universe_matrix_predictions = torch.from_numpy(batch_span_universe_matrix_predictions)

                if batch_span_universe_matrix_predictions.dtype == torch.float16:
                    # some operator that we need are not implemented for half
                    batch_span_universe_matrix_predictions = batch_span_universe_matrix_predictions.float()

                not_inf = \
                    batch_span_universe_matrix_predictions[batch_span_universe_matrix_predictions != -math.inf]

                if torch.numel(not_inf) > 0:
                    act_min_score = torch.min(not_inf).item()

                    if act_min_score < min_score:
                        min_score = act_min_score

                    act_max_score = torch.max(not_inf).item()
                    if act_max_score > max_score:
                        max_score = act_max_score

                for batch_offset, gt_span_universes in enumerate(batch.gt_span_universes):
                    gt_spans = self.spans(gt_span_universes.cpu().numpy(), goes_out=False)

                    if len(gt_spans) > 0:
                        span_len = max(e - s for _, s, e in gt_spans)

                        span_lengths[span_len] += 1

                self._predictions[-1].append(batch_span_universe_matrix_predictions)
                self._batches[-1].append(batch)

            ext_keyphrases_per_doc[len(self._ground_truths[doc_i])] += 1
            self._predictions.append([])
            self._batches.append([])

        assert min_score != math.inf
        assert max_score != -math.inf

        print("Spans lengths:")
        print_buckets_histogram(span_lengths)
        print("Extractive keyphrases per document:")
        print_buckets_histogram(ext_keyphrases_per_doc)

        print("Search space:")
        max_span_size_space = sorted(span_lengths.keys()) if self.max_span_size is None else [self.max_span_size]
        if self.n_best is None:
            n_best_space = sorted(ext_keyphrases_per_doc.keys())
            if n_best_space[0] == 0:
                n_best_space = n_best_space[1:]
        else:
            n_best_space = [self.n_best]

        print(f"\tmax_span_size:\t{max_span_size_space}")
        if self.score_threshold is None:
            print(f"\tscore_threshold:\t[{min_score},{max_score}]")
        else:
            print(f"\tscore_threshold:\t[{self.score_threshold}]")
        print(f"\tn_best:\t{n_best_space}", flush=True)

        if self.score_threshold is not None and len(max_span_size_space)*len(n_best_space) <= self._trials:
            # is not too big space for a grid search
            best_combination = self.grid_search(max_span_size_space, [self.score_threshold], n_best_space)
        else:
            wrapper = FilterWrapperForHyperTune(self.eval)

            space = {
                "max_span_size":
                    hyperopt.hp.choice("max_span_size", max_span_size_space),
                "score_threshold":
                    hyperopt.hp.uniform("score_threshold", min_score, max_score)
                    if self.score_threshold is None else hyperopt.hp.choice("score_threshold", [self.score_threshold]),
                "n_best": hyperopt.hp.choice("n_best", n_best_space),
            }
            trials = hyperopt.Trials()
            hyperopt.fmin(wrapper, space, algo=hyperopt.tpe.suggest, max_evals=self._trials, trials=trials)
            best_trial_result = trials.best_trial["result"]
            best_combination = best_trial_result["config"]
            best_combination = best_combination["max_span_size"], best_combination["score_threshold"], \
                               best_combination["n_best"], -best_trial_result["loss"]

        self._predictions = None
        self._ground_truths = None
        self._batches = None

        return best_combination

    @staticmethod
    @njit
    def spans(universe_matrix: np.array, goes_out: Union[float, bool] = -math.inf) -> List[Tuple[float, int, int]]:
        """
        Converts universe matrix to spans.

        :param universe_matrix: the universe matrix for conversion
        :param goes_out: values that signalises that given span should be filtered out
        :return: list of spans
            a span is defined by its score and token level offsets
        """
        spans = []

        valid_spans = np.argwhere(universe_matrix != goes_out)

        for indices in valid_spans:
            spans.append((universe_matrix[indices[0], indices[1]], indices[0], 1 + indices[1]))

        return spans

    def eval(self, combination: Tuple[int, float, int]) -> float:
        """
        Performs evaluation of filters combination.

        :param combination: (max_span_size, score_threshold, n_best)
        :return: score
        """

        act_filter = CompoundFilter([
            SpanSizeFilter(combination[0]),
            ThresholdFilter(combination[1])
        ])

        filtered_predictions = []

        for documents_matrices, documents_batches in zip(self._predictions, self._batches):
            documents_spans = TopKeeper(combination[2])

            for batch_u_matrix, batch in zip(documents_matrices, documents_batches):
                for sample_spans, sample_spans_scores in \
                        Model.batch_2_spans(batch_u_matrix, batch, act_filter, combination[2]):
                    for span, score in zip(sample_spans, sample_spans_scores):
                        documents_spans.push(score, span)

            filtered_predictions.append(set(documents_spans))

        metric = self.metric.eval(filtered_predictions, self._ground_truths)

        return metric["lower case lemma"][2]

    def grid_search(self, max_span_size: List[int], score_threshold: List[float], n_best: List[int], ) \
            -> Tuple[int, float, int, float]:
        """
        Finds the best combination.
        :param max_span_size: list of parameters that should be tried
        :param score_threshold: list of parameters that should be tried
        :param n_best: list of parameters that should be tried
        :return: best combination (max_span_size, score_threshold, n_best, F1)
        """

        combinations: List[Tuple[int, float, int]] = list(itertools.product(max_span_size, score_threshold, n_best))

        evaluations = []

        workers_names_queue = Queue()
        for t_id in range(len(self._use_devices)):
            workers_names_queue.put(str(t_id))

        def initializer(q: Queue):
            worker_name = q.get()
            threading.current_thread().name = worker_name

        with ThreadPool(len(self._use_devices), initializer=initializer, initargs=(workers_names_queue,)) as pool:
            iter_over = tqdm(pool.imap(self.eval, combinations), total=len(combinations),
                             desc="Finding best filters combination",
                             unit="combination")
            for e in iter_over:
                evaluations.append(e)
                iter_over.set_description(f"Finding best filters combination ({max(evaluations)})")

        max_i = torch.argmax(torch.tensor(evaluations)).item()
        return combinations[max_i] + (evaluations[max_i],)