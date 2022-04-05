# -*- coding: UTF-8 -*-
""""
Created on 30.03.21
Model for keywords extraction task. This model is using assumption that P(start) and P(end) offsets of a
    span are independent, thus:
        P(start,end) = P(start)P(end)

:author:     Martin Dočekal
"""
from typing import Optional, Tuple

import torch

from .model import Model
from ..batch import Batch
from ..training.optimizer_factory import OptimizerFactory
from ..training.scheduler_factory import SchedulerFactory


class Independent(Model):
    """
    Model for keywords extraction task. This model is using assumption that P(start) and P(end) offsets of a
    span are independent, thus:
        P(start,end) = P(start)P(end)

    Visualization of model:

         start          end
          |              |
        ---------------------
        |      OUR NN       |
        ---------------------
               ...|...        <- we are running our NN on each hidden state separately
        | | | | | | | | | | | <- hidden states
        ---------------------
        |    transformer    |
        ---------------------
    """

    def __init__(self, transformer: str, cache_dir: Optional[str] = None, local_files_only: bool = False,
                 n_best: Optional[int] = None, max_span_size: Optional[int] = None,
                 score_threshold: Optional[float] = None,
                 optimizer: Optional[OptimizerFactory] = None, scheduler: Optional[SchedulerFactory] = None,
                 init_pretrained_transformer_weights: bool = True, online_weighting: bool = True,
                 freeze_transformer: bool = False):

        super().__init__(transformer, cache_dir, local_files_only, n_best, max_span_size, score_threshold, optimizer,
                         scheduler, init_pretrained_transformer_weights, online_weighting,
                         freeze_transformer)

        self.start_end_projection = torch.nn.Linear(self.transformer.config.hidden_size, 2)
        self.init_as_transformer_weights()

    def loss(self, outputs: Tuple[torch.Tensor, torch.Tensor], gt_span_universe: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss for given output and ground truths.

        :param outputs: expects tuple (predicted start scores, predicted end scores)
        :param gt_span_universe: ground truth universe matrix
            Span universe boolean matrix for an input sequence of size N is a matrix U of size NxN.
            Where each span score is identified by its start and end token indices.
        :return: loss value
        """
        starts = torch.any(gt_span_universe, 2).float()
        ends = torch.any(gt_span_universe, 1).float()
        return self.criterion(outputs[0], starts) + self.criterion(outputs[1], ends)

    def outputs_2_span_scores(self, outputs: Tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor) \
            -> torch.Tensor:
        """
        Converts outputs from a model a score for each span in span universe matrix. (batch wise)

        Span universe matrix for an input sequence of size N is a matrix U of size NxN. Where each span score is
        identified by its start and end token indices.

        E.G. Element U[1][3] is score for span that starts on token with index 1 and ends on token with index 3.

        Each score x is convertible to probability by following formula:
            e^x

        :param outputs: Expects tuple (start scores, end scores)
        :param attention_mask: Mask to avoid performing attention on padding token indices. 1 NOT  MASKED, 0 for MASKED.
        :return: universe matrix
            Size: BATCH x INPUT_SEQUENCE_LEN x INPUT_SEQUENCE_LEN
        """
        # The probability is
        #   P(start,end) = P(start)P(end)
        # As the activation function we are using sigmoid.
        #   P(start,end) = sigmoid(s) * sigmoid(f)
        # Where s is start score and f is end score.
        # We need score j for that holds:
        #   P(start,end) = e^j = sigmoid(s) * sigmoid(f)
        # So let's use ln:
        #   ln e^j = ln(sigmoid(s) * sigmoid(f))
        #   j = ln(sigmoid(s)) + ln(sigmoid(f))

        universe_matrix = self.log_sigmoid(outputs[0]).unsqueeze(2) + self.log_sigmoid(outputs[1]).unsqueeze(1)
        return self.mask_impossible_spans(universe_matrix, attention_mask)

    def use_model(self, batch: Batch) -> Tuple[torch.Tensor, ...]:
        return self(batch.tokens, batch.attention_masks)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward phase on the model.

        :param input_ids: parsed tokens
        :param attention_mask: Mask to avoid performing attention on padding token indices. 1 NOT  MASKED, 0 for MASKED.
        :return:
            start scores BATCH X SEQUENCE SIZE
            end scores BATCH X SEQUENCE SIZE
        """

        transformer_outputs = self._pass_to_transformers(input_ids, attention_mask)

        scores = self.start_end_projection(transformer_outputs)  # the size of the result is: BATCH X SEQUENCE SIZE X 2

        return scores[:, :, 0], scores[:, :, 1]
