# -*- coding: UTF-8 -*-
""""
Created on 29.03.21
Contains basic abstract class for models.

:author:     Martin Dočekal
"""
import logging
import math
from abc import abstractmethod
from contextlib import nullcontext
from typing import Optional, Union, Tuple, List, Generator

import torch
from pytorch_lightning import LightningModule
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import _LRScheduler  # TODO: protected member access, seems dirty :(
from transformers import AutoModel, AutoConfig, PreTrainedModel

from qbek.batch import Batch
from qbek.predictions_filters import CompoundFilter, SpanSizeFilter, ThresholdFilter, \
    PredictionsFilter
from qbek.training.optimizer_factory import OptimizerFactory
from qbek.training.scheduler_factory import SchedulerFactory


class OwnBCEWithLogitsLoss:
    """
    By the time this is written separate use of sigmoid and BCELoss is  mandatory, because
    BCEWithLogitsLoss is returning nan for -math.inf. It is known issue on the github:
      https://github.com/pytorch/pytorch/issues/49844
    """

    def __init__(self, online_weighting: bool = True, reduction: str = "mean"):
        """
        Prepares bce loss so it will be instantiated only once.

        :param online_weighting: If true then the online calculated (on batch lvl)
            weight for positive class will be used.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """

        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.online_weighting = online_weighting

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Applies sigmoid on inputs and uses BCELoss.

        :param inputs: (N,∗) where ∗*∗ means, any number of additional dimensions
        :param targets: (N,∗)(N, *)(N,∗) , same shape as the input
        :return: loss
        """

        # we are handling the -inf problem here, because inputs are going to sigmoid anything <-100 will be zero
        # and anything>100 will be one
        inputs = inputs.clamp(-100, 100)

        with autocast(enabled=False):
            # we need to disable autocast (half precision) because
            # torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
            targets = targets.float()

            if self.online_weighting:
                pos_weight = torch.tensor(1.0, device=inputs.device)
                num_ones = torch.sum(targets)  # targets is tensor of zeros and ones
                if num_ones > 0:
                    zeros_mask = targets == 0.0
                    num_zeros = torch.sum(zeros_mask)

                    pos_weight = num_zeros / num_ones

                self.bce.pos_weight = pos_weight

            return self.bce(inputs.float(), targets)


class Model(LightningModule):
    """
    Abstract base class for models.

    :ivar pred_span_universe_matrix: True: predict method returns span universe matrix and batch.
        False: predict method returns List of predicted spans in form of tuple:
                (spans, scores)
            for each sample in the batch.
            A span is tuple (start char offset, end char offset)
        Default is False.
    :vartype pred_span_universe_matrix: bool
    """

    @abstractmethod
    def __init__(self, transformer: str, cache_dir: Optional[str] = None, local_files_only: bool = False,
                 n_best: Optional[int] = None, max_span_size: Optional[int] = None,
                 score_threshold: Optional[float] = None,
                 optimizer: Optional[OptimizerFactory] = None, scheduler: Optional[SchedulerFactory] = None,
                 init_pretrained_transformer_weights: bool = True,
                 online_weighting: bool = True, freeze_transformer: bool = False):
        """
        Initialization of new extractive_reader with config.

        :param transformer: transformer model that will be used (see https://huggingface.co/models)
        :param cache_dir: Cache where the transformers library will save the models.
            Use when you want to specify concrete path.
        :param local_files_only: If true only local files in cache will be used to lead models and tokenizer
        :param n_best: Only N best (the ones with biggest score) spans are predicted.
        :param max_span_size: No predicted span will have greater size (number of sub-word tokens) than this.
        :param score_threshold: For all predicted spans holds that scores must be >= than this threshold.
        :param optimizer: Factory for creating optimizer for training.
        :param scheduler: Factory for creating learning rate scheduler for training.
        :param init_pretrained_transformer_weights: Uses pretrained weights for transformer part.
            If False uses random initialization.
        :param online_weighting: If true then the online calculated (on batch lvl)
            weight for positive class will be used in loss function.
        :param freeze_transformer: Freezes the transformer part. (not the input embeddings)
        :raise AttributeError: When the scheduler is passed without optimizer.
        """
        super().__init__()

        try:
            if init_pretrained_transformer_weights:
                self.transformer = AutoModel.from_pretrained(transformer, cache_dir=cache_dir,
                                                             local_files_only=local_files_only)
            else:
                self.transformer = AutoModel.from_config(
                    AutoConfig.from_pretrained(transformer, cache_dir=cache_dir, local_files_only=local_files_only))
        except ValueError:
            # try again, might be connection error
            if init_pretrained_transformer_weights:
                self.transformer = AutoModel.from_pretrained(transformer, cache_dir=cache_dir, local_files_only=True)
            else:
                self.transformer = AutoModel.from_config(
                    AutoConfig.from_pretrained(transformer, cache_dir=cache_dir, local_files_only=True))

        if optimizer is None and scheduler is not None:
            raise AttributeError("Scheduler without optimizer can not be used.")

        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.log_sigmoid = torch.nn.LogSigmoid()
        self._n_best = n_best
        self._max_span_size = max_span_size
        self._score_threshold = score_threshold
        self._pred_filter = None
        self._apply_filters()
        self.pred_span_universe_matrix = False
        self.hparams.update({
            "transformer": transformer,
            "n_best": n_best,
            "max_span_size": max_span_size,
            "score_threshold": score_threshold
        })
        self.criterion = OwnBCEWithLogitsLoss(online_weighting=online_weighting)

        self.freeze_transformer = freeze_transformer

    @property
    def n_best(self) -> Optional[int]:
        """
        Only N best (the ones with biggest score) spans are predicted per a document sample.
        If none this kind of filter is not applied.
        """

        return self._n_best

    @n_best.setter
    def n_best(self, n: Optional[int]):
        """
        Sets new value for n best prediction filter.

        :param n: The new n. If none filter is not used.
        """
        self._n_best = n
        self.hparams["n_best"] = n

    @property
    def max_span_size(self) -> Optional[int]:
        """
        No predicted span will have greater size (number of sub-word tokens) than this.
        If none this kind of filter is not applied.
        """

        return self._max_span_size

    @max_span_size.setter
    def max_span_size(self, s: int):
        """
        Sets new value for max span size prediction filter.

        :param s: The new maximum size. If none filter is not used.
        """
        self._max_span_size = s
        self.hparams["max_span_size"] = s
        self._apply_filters()

    @property
    def score_threshold(self) -> Optional[int]:
        """
        For all predicted spans holds that scores must be >= than this threshold.
        If none this kind of filter is not applied.
        """

        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, t: float):
        """
        Sets new value for score threshold prediction filter.

        :param t: The new threshold. If none filter is not used.
        """
        self._score_threshold = t
        self.hparams["score_threshold"] = t
        self._apply_filters()

    def _pass_to_transformers(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes inputs to transformer

        :param input_ids: parsed tokens
        :param attention_mask: Mask to avoid performing attention on padding token indices. 1 NOT  MASKED, 0 for MASKED.
        :return: transformed sequence
        """

        with torch.no_grad() if self.freeze_transformer else nullcontext():
            transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)

        return transformer_outputs[0]

    def _apply_filters(self):
        """
        Sets the pred_filter according to variables _max_span_size and _score_threshold.
        """

        filters = []

        if self._max_span_size is not None:
            filters.append(SpanSizeFilter(self._max_span_size))

        if self._score_threshold is not None:
            filters.append(ThresholdFilter(self._score_threshold))

        self._pred_filter = CompoundFilter(filters)

    def init_as_transformer_weights(self):
        """
        Use the same initializer for linear projections as for the transformer was used.
        """

        for ch in self.children():
            if issubclass(ch.__class__, torch.nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: self.transformer.__class__._init_weights(self.transformer, module))

    @abstractmethod
    def loss(self, outputs, gt_span_universe: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss for model's output.

        :param outputs: outputs from model
        :param gt_span_universe: ground truth universe matrix
            Span universe boolean matrix for an input sequence of size N is a matrix U of size NxN.
            Where each span is identified by True on its start and end token indices.
        :return: loss value
        """
        pass

    @abstractmethod
    def outputs_2_span_scores(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Converts outputs from a model a score for each span in span universe matrix. (batch wise)

        Span universe matrix for an input sequence of size N is a matrix U of size NxN. Where each span score is
        identified by its start and end token indices.

        E.G. Element U[1][3] is score for span that starts on token with index 1 and ends on token with index 3.

        Each score x is convertible to probability by following formula:
            e^x

        :param outputs: Outputs from model.
        :param attention_mask: Mask to avoid performing attention on padding token indices. 1 NOT  MASKED, 0 for MASKED.
        :return: universe matrix
            Size: BATCH X INPUT_SEQUENCE_LEN x INPUT_SEQUENCE_LEN
        """
        pass

    @staticmethod
    def batch_2_spans(universe_matrix: torch.Tensor, batch: Batch, u_mat_filter: Optional[PredictionsFilter],
                      n_best: int) -> Generator[Tuple[List[str], List[float]], None, None]:
        """
        Extracts spans from given batch.

        :param universe_matrix: Universe matrix of all span scores for single sample.
        :param batch: Batch containing samples of universe matrix.
        :param u_mat_filter: Filter that should be applied to the universe matrix.
        :param n_best: How many unique keyphrases should be extracted for one sample.
        :return: Predicted spans in form of tuple:
                (spans, scores)
            for each sample in the batch.
            A span is string
            with scores
        """

        if u_mat_filter is not None:
            universe_matrix = u_mat_filter(universe_matrix)

        scores, indices = torch.sort(universe_matrix.view(universe_matrix.shape[0], -1), descending=True)

        for i_sample, (sorted_scores, sorted_indices) in enumerate(zip(scores, indices)):

            keyphrases_2_score = {}

            mask = sorted_scores > -math.inf

            sorted_scores = sorted_scores[mask].tolist()
            sorted_start_token = (sorted_indices[mask] // universe_matrix.shape[2]).tolist()
            sorted_end_token = (sorted_indices[mask] % universe_matrix.shape[2]).tolist()

            for score, i_start_token, i_end_token in zip(sorted_scores, sorted_start_token, sorted_end_token):
                try:
                    span = batch.get_span(i_sample, i_start_token, i_end_token)
                    if span in keyphrases_2_score:
                        keyphrases_2_score[span] = max(keyphrases_2_score[span], score)
                    else:
                        keyphrases_2_score[span] = score

                    if len(keyphrases_2_score) == n_best:
                        break
                except IndexError:
                    # Special token or token out of sequence was indexed
                    pass

            top_keyphrases = []
            top_keyphrases_scores = []
            for k, s in keyphrases_2_score.items():
                top_keyphrases.append(k)
                top_keyphrases_scores.append(s)

            yield top_keyphrases, top_keyphrases_scores

    def predictions(self, universe_matrix: torch.Tensor, batch: Batch) \
            -> Generator[Tuple[List[str], List[float]], None, None]:
        """
        From universe_matrix containing scores and batch creates predictions.
        Applies filters.

        :param universe_matrix: Universe matrix of all span scores for single sample.
        :param batch: Batch containing samples of universe matrix.
        :return: Predicted spans in form of tuple:
                (spans, scores)
            for each sample in the batch.
            A span is string
            with scores
        """
        universe_matrix = universe_matrix

        yield from self.batch_2_spans(universe_matrix, batch, self._pred_filter, self.n_best)

    @staticmethod
    def mask_impossible_spans(universe_matrix: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Masks impossible spans (lower triangular without diagonal) with -inf. Works inplace.

        :param universe_matrix: Matrix of scores for each span in sequence.
        :param attention_mask: Mask to avoid performing attention on padding token indices. 1 NOT  MASKED, 0 for MASKED.
        :return: Works inplace, but also returns the matrix.
        """
        universe_matrix.masked_fill_(
            torch.tril(
                torch.ones(universe_matrix.shape[1], universe_matrix.shape[2], dtype=torch.bool,
                           device=universe_matrix.device),
                diagonal=-1),
            float("-inf"))

        mask = attention_mask == 0
        universe_matrix[mask.unsqueeze(1).expand_as(universe_matrix)] = float("-inf")

        return universe_matrix

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer],
                                                                         List[_LRScheduler]]]:
        """
        Optimizers and schedulers that should be used during the training process.

        :return: optimizers and schedulers
        :raise RuntimeError: When the optimizer is not available.
        """

        if self.optimizer_factory is None:
            raise RuntimeError("No optimizer. Did you pass optimizer to the init method?")

        optimizer = self.optimizer_factory.create(self)

        if self.scheduler_factory is None:
            return optimizer
        else:
            return [optimizer], [self.scheduler_factory.create(optimizer)]

    @abstractmethod
    def use_model(self, batch: Batch) -> Tuple[torch.Tensor, ...]:
        """
        Uses model on given batch

        :param batch: Batch that should be passed to the model.
        :return: Should return components of tuple that can be passed to the loss function.
        """
        pass

    def training_step(self, batch: Batch, batch_idx) -> torch.Tensor:
        """
        Performs single training step with given batch.

        :param batch: Batch that should be passed to the model.
        :param batch_idx: batch id
        :return: loss of this step
        """
        assert torch.isnan(batch.gt_span_universes).any() == False
        assert batch.gt_span_universes.dtype == torch.bool

        loss = self.loss(self.use_model(batch), batch.gt_span_universes)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: Optional[int] = None) -> \
            Union[Tuple[torch.Tensor, Batch], List[Tuple[List[str], List[float], int]]]:
        """
        Predictions for a batch.

        :param batch: batch for prediction
        :param batch_idx: batch id
        :param dataloader_idx: i was not able to learn what is that
        :return: span universe matrices for given batch and the batch itself if self.pred_span_universe_matrix is True
            else
                List of predicted spans in form of tuple:
                    (spans, scores, line offset)
                for each model sample in the batch.
                A span is string.
        """
        span_universe_matrix = self.outputs_2_span_scores(self.use_model(batch), attention_mask=batch.attention_masks)
        if self.pred_span_universe_matrix:
            return span_universe_matrix.cpu(), batch.to(torch.device("cpu"))
        else:
            return [
                (spans, scores, batch.line_offsets[i])
                for i, (spans, scores) in enumerate(self.predictions(span_universe_matrix, batch))
            ]

    def validation_step(self, batch: Batch, batch_idx) -> torch.Tensor:
        """
        Performs single validation step with given batch.

        :param batch: Batch that should be passed to the model.
        :param batch_idx: batch id
        :return: loss of this step
        """
        loss = self.loss(self.use_model(batch), batch.gt_span_universes)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs: List[torch.tensor]):
        loss = sum(o.item() for o in outputs) / len(outputs)
        logging.info(f"Validation loss in epoch {self.current_epoch} and step {self.global_step} is {loss}.")
