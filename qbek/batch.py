# -*- coding: UTF-8 -*-
""""
Created on 04.04.2021
Module containing class that represents a batch.

:author:     Martin Dočekal
"""
from typing import List, Tuple, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class Batch:
    """
    Representation of device batch.
    # line_offset, input_string, tokens, tokens_offset_mapping, spans, document embedding
    """

    def __init__(self, tokens: torch.Tensor, attention_masks: torch.Tensor, gt_span_universes: Optional[torch.Tensor],
                 line_offsets: List[int], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        """
        Batch initialization

        :param tokens: Tokens that can be used as model input.
        :param attention_masks: Attention mask for tokens.
        :param gt_span_universes: Span universes for each model sample.
        :param line_offsets: line offset of the original documents
        :param tokenizer: Tokenizer that wa used for input tokenization
            is there just for decoding the span to string representation
            WARNING: two batches with different tokenizer are equal
        """

        self.tokens = tokens
        self.attention_masks = attention_masks
        self.gt_span_universes = gt_span_universes
        self.line_offsets = line_offsets
        self._tokenizer = tokenizer
        self._spacial_tokens = {tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id}

    def to(self, device: torch.device) -> "Batch":
        """
        Moves batch's values to given device.

        :param device: the device this batch's values will be moved to
        :return: batch with moved values
        """

        return Batch(self.tokens.to(device),
                     self.attention_masks.to(device),
                     None if self.gt_span_universes is None else self.gt_span_universes.to(device),
                     self.line_offsets, self._tokenizer)

    def get_span(self, sample: int, start: int, end: int) -> str:
        """
        Get span for given sample and boundary token indices.

        :param sample: sample offset in batch
        :param start: offset of the start token
        :param end: offset of the end token
        :return: string that represents selected span
        :raises IndexError: when invalid indices are provided or span contains sep, cls or pad tokens
        """
        if end < start:
            raise IndexError("End token offset points before start token.")
        span = self.tokens[sample][start:end+1].tolist()

        if any(token in self._spacial_tokens for token in span):
            raise IndexError(f"The selected span '{span}' contains sep, cls or pad token.")

        return self._tokenizer.decode(span)

    def __len__(self) -> int:
        """
        batch size
        """
        return len(self.line_offsets)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return torch.equal(self.tokens, other.tokens) \
                   and torch.equal(self.attention_masks, other.attention_masks) \
                   and torch.equal(self.gt_span_universes, other.gt_span_universes) \
                   and self.line_offsets == other.line_offsets
        return False

    def split(self, offset: int) -> Tuple["Batch", "Batch"]:
        """
        Splits batch into two.

        :param offset: The first offset of the second batch.
        :return: the two parts of the original batch
        """

        first = Batch(tokens=self.tokens[:offset],
                      attention_masks=self.attention_masks[:offset],
                      gt_span_universes=None if self.gt_span_universes is None else self.gt_span_universes[:offset],
                      line_offsets=self.line_offsets[:offset],
                      tokenizer=self._tokenizer)

        second = Batch(tokens=self.tokens[offset:],
                       attention_masks=self.attention_masks[offset:],
                       gt_span_universes=None if self.gt_span_universes is None else self.gt_span_universes[offset:],
                       line_offsets=self.line_offsets[offset:],
                       tokenizer=self._tokenizer)

        return first, second
