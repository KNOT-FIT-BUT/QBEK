# -*- coding: UTF-8 -*-
""""
Created on 11.04.21

:author:     Martin Dočekal
"""
import unittest
from unittest import TestCase

import torch
from typing import List, Optional, Tuple, Iterable

from qbek.lemmatizer import Lemmatizer


class MockTransformer(torch.nn.Module):
    """
    Mockup for transformer

    :ivar output_dim: Output dimension of a transformer representation for a single input token.
    :vartype output_dim: int
    :ivar outputs_hist: All generated tensors in chronological order.
    :vartype outputs_hist: List[torch.Tensor]
    :ivar test_case: If not None than generates always this tensor as output.
    :vartype test_case: Optional[TestCase]
    :ivar always_same_tensor: If not None than generates always this tensor as output.
    :vartype always_same_tensor: Optional[torch.Tensor]
    """

    def __init__(self, output_dim: int, test_case: Optional[TestCase] = None,
                 always_same_tensor: Optional[torch.Tensor] = None):
        """
        Initializes the transformer.

        :param output_dim: Output dimension of a transformer representation for a single input token.
        :param test_case: If not None than there will be additional controls.
        :param always_same_tensor: If not None than generates always this tensor as output.
        """
        super().__init__()
        self.output_dim = output_dim
        self.outputs_hist = []
        self.test_case = test_case
        self.always_same_tensor = always_same_tensor

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs) -> List[torch.Tensor]:
        """
        Generates random tensor or always same tensor.

        :param input_ids: Ids to get appropriate shape of output tensor.
        :param attention_mask: Monitors that attention mask was passed.
        :return: Single element list with random tensor.
        """
        if self.test_case is not None:
            self.test_case.assertIsNotNone(attention_mask)
            self.test_case.assertEqual(input_ids.shape, attention_mask.shape)

        if self.always_same_tensor is None:
            out = torch.rand(input_ids.shape + (self.output_dim,))
        else:
            out = self.always_same_tensor.clone()

        self.outputs_hist.append(out)

        return [self.outputs_hist[-1]]


class MockForward:
    """
    Mock of forward pass for model.

    :ivar expected_input_ids: Input ids that are expected to be passed to forward method.
    :vartype expected_input_ids: torch.Tensor
    :ivar expected_attention_mask: Attention_mask that is expected to be passed to forward method.
    :vartype expected_attention_mask: torch.Tensor
    :ivar expected_output: This output will be returned
    :vartype expected_output: Tuple[torch.Tensor, torch.Tensor]
    :ivar test_case: TestCase that wil be used for additional checks.
    :vartype test_case: unittest.TestCase
    :ivar called: True if this functor was called at least once.
    :vartype called: bool
    """

    def __init__(self, expected_input_ids: torch.Tensor, expected_attention_mask: torch.Tensor,
                 expected_output: Tuple[torch.Tensor, torch.Tensor], expected_doc_embeds: Optional[torch.Tensor],
                 test_case: unittest.TestCase):
        """
        Initialization of functor.

        :param expected_input_ids: Input ids that are expected to be passed to forward method.
        :param expected_attention_mask: Attention_mask that is expected to be passed to forward method.
        :param expected_output: This output will be returned
        :param expected_doc_embeds: Document embeddings that are expected.
        :param test_case: TestCase that wil be used for additional checks.
        """
        super().__init__()
        self.expected_input_ids = expected_input_ids
        self.expected_attention_mask = expected_attention_mask
        self.expected_doc_embeds = expected_doc_embeds
        self.test_case = test_case
        self.expected_output = expected_output
        self.called = False

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 doc_embeds: Optional[torch.Tensor] = None):
        """
        Call of forward pass.

        :param input_ids: token ids
        :param attention_mask: mask for attention
        :param doc_embeds: Document embedding that should be added to the positional and word embeddings.
            Don't forget to activate them in constructor.
        :return: returns expected_output
        """
        self.test_case.assertListEqual(self.expected_input_ids.tolist(), input_ids.tolist())
        self.test_case.assertListEqual(self.expected_attention_mask.tolist(), attention_mask.tolist())

        if self.expected_doc_embeds is None:
            self.test_case.assertIsNone(doc_embeds)
        else:
            self.test_case.assertListEqual(self.expected_doc_embeds.tolist(), doc_embeds.tolist())

        self.called = True
        return self.expected_output


class MockDumbLemmatizer(Lemmatizer):
    """
    This lemmatizer leaves only alpha characters.
    """

    def lemmatize(self, words: Iterable[str]) -> Tuple[str, ...]:
        return tuple(words)


class MockLastCapLemmatizer(Lemmatizer):
    """
    This lemmatizer removes last character if it is capitalized and is not the only character.
    """

    def lemmatize(self, words: Iterable[str]) -> Tuple[str, ...]:
        return tuple("".join(w[:-1] if len(w) > 1 and w[-1].isupper() else w for w in words))
