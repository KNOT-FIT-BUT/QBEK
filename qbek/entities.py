# -*- coding: UTF-8 -*-
""""
Created on 25.01.21
This module contains representations of entities. Entities like context, keyphrase span, etc.

:author:     Martin Dočekal
"""
import enum
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Set, Iterable, Dict, Optional, Tuple

import spacy
from spacy.lang.cs import Czech
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.lang.fr import French

from qbek.lemmatizer import SpaCyLemmatizer, MorphoditaLemmatizer, Lemmatizer, LemmatizerFactory, \
    MorphoditaLemmatizerFactory, SpaCyLemmatizerFactory, DummyLemmatizer, PorterStemmer, PorterStemmerFactory
from qbek.tokenizer import Tokenizer, SpaCyTokenizer, AlphaTokenizer


@dataclass
class Keyphrase:
    """
    Representation of keyphrase span.
    """

    orig_context: str  # context of keyphrase
    start_offset: int  # start character offset in original context of keyphrase
    end_offset: int  # end character offset in original context of keyphrase

    def __repr__(self):
        """
        Format of representation is:
            keyphrase \t context \t start_offset end_offset
        """

        return f"{str(self)}\t{self.orig_context}\t{self.start_offset} {self.end_offset}"

    def __str__(self):
        return self.orig_context[self.start_offset:self.end_offset]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.orig_context == other.orig_context and self.start_offset == other.start_offset \
                   and self.end_offset == other.end_offset
        return False

    def __hash__(self):
        return hash((self.orig_context, self.start_offset, self.end_offset))


@dataclass
class Context:
    """
    Representation of a context for keyphrase extraction/generation.
    """
    text: str  # it is expected that the context does not contain \t
    keyphrases: List[Keyphrase]

    @classmethod
    def from_str(cls, s: str) -> "Context":
        """
        Creates Context from it's string representation.

        Example:
            Context.from_str("I am a keyphrase\t7 16")

        :param s: String representation of a context.
        :type s: str
        :return: parsed context
        :rtype: Context
        """
        parts = s.split("\t")
        keyphrases = []

        if len(parts) > 1:
            for offsets in parts[1:]:
                start_offset, end_offset = offsets.split()
                keyphrases.append(Keyphrase(parts[0], int(start_offset), int(end_offset)))

        return Context(parts[0], keyphrases)

    def __repr__(self):
        return str(self)

    def __str__(self):
        """
        Format:
            "EXTRACTED CONTEXT"\tSTART_OF_KEYWORD_CHARACTER_OFFSET END_OF_KEYWORD_CHARACTER_OFFSET...
        """
        res = self.text
        for k in self.keyphrases:
            res += f"\t{k.start_offset} {k.end_offset}"

        return res

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.text == other.text and set(self.keyphrases) == set(other.keyphrases)
        return False

    def __hash__(self):
        return hash((self.text,) + tuple(self.keyphrases))


class StringableDocument(ABC):
    """
    Abstract base class for documents that supports conversion to str.
    """

    tokenizer = AlphaTokenizer()
    shared_lemmatizer = DummyLemmatizer()
    """If not set to none will be used as lemmatizer for a document as long another one is not passed to init."""

    def __init__(self, uuid: str, title_statement: str, contexts: List[Context], annotated_keyphrases: Iterable[str],
                 lemmatizer: Optional[Lemmatizer] = None):
        """
        Base initialization of stringable document.

        :param uuid: Unique identifier used by library.
        :param title_statement: the query field
        :param contexts: parsed contexts
        :param annotated_keyphrases: Annotated keyphrases for this document.
        :param lemmatizer: Lammatizer that should be used.
        """

        self._uuid = uuid
        self._title_statement = title_statement
        self._contexts = contexts
        self.annotated_keyphrases = set(annotated_keyphrases)

        if self.shared_lemmatizer is not None and lemmatizer is None:
            self.lemmatizer = self.shared_lemmatizer
        else:
            self.lemmatizer = lemmatizer

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def title_statement(self) -> str:
        return self._title_statement

    @property
    def ext_kp(self) -> Set[str]:
        """
        Keyphrases from contexts (the searched one).
        """

        keyphrases = set()

        for c in self.contexts:
            for k in c.keyphrases:
                keyphrases.add(str(k))

        return keyphrases

    @property
    def annotated_ext_kp(self) -> Set[str]:
        """
        Returns annotated keyphrases that occurs in document
        (tokenization, lemmatization, and lower case normalization is used).
        """

        return self.annotated_keyphrases - self.gen_kp

    @property
    def gen_kp(self) -> Set[str]:
        """
        Returns keyphrases that can not be extracted from document's contexts
        (tokenization, lemmatization, and lower case normalization is used).
        """
        gen_kp = set()

        lw_ext_kp = set(self.normalize(k) for k in self.ext_kp)

        for k in self.annotated_keyphrases:
            if self.normalize(k) not in lw_ext_kp:
                gen_kp.add(k)

        return gen_kp

    @property
    def contexts(self) -> List[Context]:
        return self._contexts

    @contexts.setter
    def contexts(self, contexts: List[Context]):
        self._contexts = contexts

    def __repr__(self):
        return str(self)

    def __str__(self):
        """
        converts to json format
        """

        goes_out = OrderedDict([
            ("uuid", self.uuid),
            ("query", self.title_statement),
            ("keyphrases", list(sorted(self.annotated_keyphrases))),
            ("contexts", [str(context) for context in self.contexts])
        ])
        return json.dumps(goes_out, ensure_ascii=False)

    @abstractmethod
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.uuid == other.uuid and self.title_statement == other.title_statement \
                   and self.annotated_keyphrases == other.annotated_keyphrases \
                   and self.contexts == other.contexts
        return False

    def __hash__(self):
        return hash(self.uuid)

    def normalize(self, s: str) -> Tuple[str, ...]:
        """
        Performs tokenization, lemmatization, and lower case normalization.
        WARNING you need to set lemmatizer.

        :param s: string that should be normalized
        :return: normalized string
        """

        return tuple(t.lower() for t in self.lemmatizer.lemmatize(self.tokenizer(s)))


@dataclass
class Document(StringableDocument):
    """
    Representation of already created document. Representation of document for parsing phase is class:`.ParDocument`.
    """

    def __init__(self, uuid: str, title_statement: str, contexts: List[Context], annotated_keyphrases: Iterable[str]):
        """
        document initialization

        :param uuid: Unique identifier used by library.
        :param title_statement: The query field.
        :param contexts: parsed contexts
        :param annotated_keyphrases: Annotated keyphrases for this document.
        """
        super().__init__(uuid, title_statement, contexts, annotated_keyphrases)

    @classmethod
    def from_json(cls, s: str) -> "Document":
        """
        Creates Document from it's json representation.

        :param s: Json representation of a Document.
        :return: parsed document
        """

        loaded = json.loads(s)
        contexts = [Context.from_str(context_str) for context_str in loaded["contexts"]]

        return Document(loaded["uuid"], loaded["query"], contexts, loaded["keyphrases"])

    def __eq__(self, other):
        return super().__eq__(other)


class Language(enum.Enum):
    """
    Representation of a language.
    Also defines set of supported languages.
    """

    CZECH = "cze"
    ENGLISH = "eng"
    GERMAN = "ger"
    FRENCH = "fre"

    @classmethod
    def lang_2_spacy(cls) -> Dict["Language", spacy.language.Language]:
        """
        Mapping from language to its spacy language model.
        """
        return {cls.CZECH: Czech, cls.ENGLISH: English, cls.GERMAN: German, cls.FRENCH: French}

    @property
    def code(self) -> str:
        """
        Language code.
        CZECH  -> cze
        """
        return self.value

    @property
    def spacy(self) -> spacy.language.Language:
        """
        Spacy language model class.
        """
        return self.lang_2_spacy()[self]

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Tokenizer for a language.
        """

        return SpaCyTokenizer.init_shared(self.spacy)

    @property
    def lemmatizer(self) -> Lemmatizer:
        """
        Lemmatizer for a language.
        """

        if self == self.CZECH:
            # for czech the morphodite is better
            return MorphoditaLemmatizer.init_shared()
        elif self == self.ENGLISH:
            return PorterStemmer()
        else:
            return SpaCyLemmatizer.init_shared(self.spacy)

    @property
    def lemmatizer_factory(self) -> LemmatizerFactory:
        """
        Lemmatizer factory for a language.
        """

        if self == self.CZECH:
            # for czech the morphodite is better
            return MorphoditaLemmatizerFactory()
        elif self == self.ENGLISH:
            return PorterStemmerFactory()
        else:
            return SpaCyLemmatizerFactory(self.spacy)


@dataclass
class DocumentResults:
    documents_line_offset: int
    spans: List[str]
    scores: List[float]
