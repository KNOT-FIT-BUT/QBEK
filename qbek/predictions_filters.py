# -*- coding: UTF-8 -*-
""""
Created on 05.04.21
This module contains classes for filtering predictions of KeywordsExtractor.

:author:     Martin Dočekal
"""
import math
from abc import ABC, abstractmethod
from typing import List

import torch


class PredictionsFilter(ABC):
    """
    Base class for functors that serves as filters for predictions.

    All filters works with span universe matrix and they are setting -inf score on position of filtered spans.
    """

    @abstractmethod
    def __call__(self, span_universe_matrix: torch.Tensor) -> torch.Tensor:
        """
        According to provided batch of span universe matrices, filters out undesired predictions.

        :param span_universe_matrix: Contains scores of all spans.
        :return: Tensor of scores where -inf is set on filtered positions.
        """
        pass


class CompoundFilter(PredictionsFilter):
    """
    This filter applies multiple filters.

    Example:
        filter = CompoundFilter([F1, F2, F3])
        Order of sub filters applications:
            F3(F2(F1(input)))
    """

    def __init__(self, filters: List[PredictionsFilter]):
        """
        Initialization of filter.

        :param filters: Filters that should by applied.
            The order of applications is determined from order in list.
        """

        self.filters = filters

    def __call__(self, span_universe_matrix: torch.Tensor) -> torch.Tensor:
        act_span_universe_matrix = span_universe_matrix

        for f in self.filters:
            act_span_universe_matrix = f(act_span_universe_matrix)

        return act_span_universe_matrix


class ThresholdFilter(PredictionsFilter):
    """
    Anything with score above or equal to given threshold passes this filter.
    """

    def __init__(self, threshold: float):
        """
        Filter initialization.

        :param threshold: Minimal score to pass this filter.
        """

        self.threshold = threshold

    def __call__(self, span_universe_matrix: torch.Tensor) -> torch.Tensor:
        res = span_universe_matrix.clone()
        res[res < self.threshold] = -math.inf
        return res


class FirstNFilter(PredictionsFilter):
    """
    First N predictions (the ones with biggest score) passes this filter.
    """

    def __init__(self, n: int):
        """
        Filter initialization.

        :param n: Number of predictions with greatest scores that should pass this filter.
        """

        self.n = n

    def __call__(self, span_universe_matrix: torch.Tensor) -> torch.Tensor:
        res = span_universe_matrix.clone()
        res_view = res.view(res.shape[0], -1)
        mask = torch.ones_like(res_view, dtype=torch.bool)
        select_top_n = res_view.shape[1] if self.n >= res_view.shape[1] else self.n
        mask.scatter_(1, torch.topk(res_view, select_top_n, 1).indices, 0)
        res_view[mask] = -math.inf
        return res


class SpanSizeFilter(PredictionsFilter):
    """
    Filters predictions of spans that haves larger size than a maximal.

    Warning filters out only the upper triangular.
    """

    def __init__(self, max_size: int):
        """
        Filter initialization.

        :param max_size: Miximal number of sub-word tokens that passes that filter.
        :type max_size: int
        """

        self.max_size = max_size

    def __call__(self, span_universe_matrix: torch.Tensor) -> torch.Tensor:
        res = span_universe_matrix.clone()
        mask = torch.triu(
            torch.ones(res.shape[1], res.shape[2], dtype=torch.bool,
                       device=res.device),
            diagonal=self.max_size)
        res.masked_fill_(mask, float("-inf"))
        return res


