# -*- coding: UTF-8 -*-
""""
Created on 21.10.21
This module contains tokenizers.

:author:     Martin Dočekal
"""
import sys
from abc import ABC, abstractmethod
from typing import Tuple, List

import spacy
from numba import njit


class Tokenizer(ABC):
    """
    Abstract class for tokenizers defining common interface.
    """

    @abstractmethod
    def __call__(self, s: str) -> Tuple[str, ...]:
        """
        Returns tokenized form of sequence.

        :param s: text sequence for tokenization
        :return: tokens
        """
        pass


class AlphaTokenizer(Tokenizer):
    """
    Class for tokenization  that separates on all non-alpha characters and removes white ones.
    """

    def __call__(self, s: str) -> Tuple[str, ...]:
        return tuple(self._tokenize(s))

    @staticmethod
    @njit
    def _tokenize(s: str) -> List[str]:
        res = []
        state = "B"  # BEGIN
        for c in s.strip():
            if c.isalpha():
                new_state = "A"  # ALPHA_BEFORE
            else:
                new_state = "N"  # NON_ALPHA_BEFORE
                if c.isspace():
                    state = "B"  # BEGIN
                    continue

            if state != new_state:
                res.append("")

            state = new_state
            res[-1] += c

        return res


class SpaCyTokenizer(Tokenizer):
    """
    Class for tokenization  that uses spaCy.
    """

    SHARED_INSTANCES = {}

    def __init__(self, lang_spacy: spacy.language.Language):
        """
        Tokenizer initialization.

        :param lang_spacy: The spacy language model.
        """
        lang_package = {
            "en": "en_core_web_sm",
            "de": "de_core_news_sm",
            "fr": "fr_core_news_sm"
        }[lang_spacy.lang]

        try:
            self.lang_stat_model = spacy.load(lang_package)
        except OSError:
            print(f"Downloading language package {lang_package} for the spaCy Tokenizer.", file=sys.stderr)

            from spacy.cli import download
            download(lang_package)
            self.lang_stat_model = spacy.load(lang_package)

    @classmethod
    def init_shared(cls, lang_spacy: spacy.language.Language) -> 'Tokenizer':
        """
        Creates shared (singleton) Tokenizer for given language model.

        :return: shared Tokenizer
        """

        if lang_spacy not in cls.SHARED_INSTANCES:
            cls.SHARED_INSTANCES[lang_spacy] = cls(lang_spacy)

        return cls.SHARED_INSTANCES[lang_spacy]

    def __call__(self, s: str) -> Tuple[str, ...]:
        return tuple(t.text for t in self.lang_stat_model(s))
