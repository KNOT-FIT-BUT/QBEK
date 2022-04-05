# -*- coding: UTF-8 -*-
""""
Created on 31.03.21
This module contains code for CzechLibrary dataset that is a dataset of keyphrases and their contexts

:author:     Martin Dočekal
"""
import functools
import json
import logging
import math
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Optional, Set, Sequence

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from windpyutils.structures.maps import ImmutIntervalMap

from qbek.batch import Batch
from qbek.datasets.preprocessing import SampleSplitter, PreprocessedDataset


class CzechLibrarySampleSplitter(SampleSplitter):
    """
    Splitting dataset samples into model samples.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """
        initialization of splitter

        :param tokenizer: Tokenizer that will be used to determine splits in order to create suitable model samples
            that will fit into a model.
        """

        self._tokenizer = tokenizer

    @staticmethod
    def join_spans(spans: Iterable[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Joins spans that haves intersection.

        :param spans: Spans that should be joined.
        :return: Set of disjunctive spans.
        """
        res = {s for s in spans}
        has_intersection = True

        while has_intersection:
            has_intersection = False

            new_res = set()
            for s, e in res:
                for in_s, in_e in new_res.copy():
                    if e >= in_s and in_e >= s:
                        has_intersection = True
                        new_res.remove((in_s, in_e))
                        new_res.add((min(s, in_s), max(e, in_e)))
                        break
                else:
                    new_res.add((s, e))
            res = new_res

        return res

    @classmethod
    def fair_split(cls, free_space: int, tokens_context_offsets: Sequence[Tuple[int, int]],
                   spans: Sequence[Tuple[int, int]]) \
            -> List[Tuple[int, int]]:
        """
        Splits tokens into sequence of roughly same size in a way that given spans will not be split.

        :param free_space: Total available free space.
        :param tokens_context_offsets: offsets for each token in sequence
        :param spans: spans that should not be split.
            Each span is tuple of character offsets (right exclusive)
        :return: List of tuples representing start character offset and end character offset (exclusive) of
            split sequences.
        :raise RuntimeError: When the sample couldn't be split.
        """
        spans = [(s, e - 1) for s, e in spans]
        # we need to join spans with intersection, because ImmutIntervalMap need disjunctive intervals
        spans_intervals = ImmutIntervalMap({s: s for s in cls.join_spans(spans)})

        parts = []

        fair_split = math.ceil(len(tokens_context_offsets) / math.ceil(len(tokens_context_offsets) / free_space))

        offset = 0
        num_of_tokens = 0
        for i, _ in enumerate(tokens_context_offsets):
            num_of_tokens += 1

            if num_of_tokens > fair_split:
                l_back = i - 1
                while l_back >= 0 and \
                        any(o in spans_intervals and spans_intervals[o][1] + 1 != tokens_context_offsets[l_back][1]
                            # it's ok if this is pointing to end
                            for o in range(tokens_context_offsets[l_back][0], tokens_context_offsets[l_back][1])):
                    l_back -= 1

                if l_back < 0 or offset > tokens_context_offsets[l_back][0]:
                    raise RuntimeError(f"Sample couldn't be split.")

                parts.append((offset, tokens_context_offsets[l_back][1]))

                offset = tokens_context_offsets[l_back + 1][0]
                num_of_tokens = i - l_back

        if num_of_tokens > 0:
            parts.append((offset, tokens_context_offsets[-1][1]))

        return parts

    def _assemble_input(self, title: str, context: str, spans: Sequence[Tuple[int, int]]) \
            -> Tuple[Optional[np.array], np.array, np.array]:
        """
        Assembles given title and context into a models input.

        :param title: Title that should be added at the start or empty string if title should not be used.
            sep token is used to separate context and title
        :param context: the context that may contain spans
        :param spans: spans that appear in given context
            the offsets are on character lvl
        :return: Result is in the form of tuple:
            tokenized prefix - tokenized prefix of input string that is same for all samples in a document (title)
            tokenized - tokenized input string that should be put on the input of a model
            keyphrase spans as INCLUSIVE token offsets
        """

        if len(title) > 0:
            input_seq = title + self._tokenizer.sep_token + context
            context_offset = len(title + self._tokenizer.sep_token)
        else:
            input_seq = context
            context_offset = 0

        spans = [(start + context_offset, end + context_offset) for start, end in spans]

        encode_res = self._tokenizer.encode_plus(input_seq, return_offsets_mapping=True,
                                                 return_attention_mask=False,
                                                 return_token_type_ids=False,
                                                 return_tensors="np")
        spans_token_lvl = []
        offset_mapping = encode_res['offset_mapping'].squeeze().astype(np.int16)
        for start, end in spans:
            spans_token_lvl.append(CzechLibrary.char_2_token_lvl_span((start, end), offset_mapping.tolist()))

        input_ids = encode_res["input_ids"].flatten().astype(np.int32)
        input_ids_prefix = None

        if len(title) > 0:
            sep_index = np.argmax(input_ids == self._tokenizer.sep_token_id)
            input_ids_prefix = input_ids[:sep_index + 1]
            input_ids = input_ids[sep_index + 1:]
        return input_ids_prefix, input_ids, np.array(spans_token_lvl, dtype=np.int16)

    def split(self, title: str, context: str) -> Iterable[Tuple[Optional[np.array], np.array, np.array]]:
        """
        Splits dataset sample into parts (model samples) which fit into a model.

        :param title: Title of document
            WARNING: if the title is too long that no space is left for a context this method doesn't split the context
            at all.
        :param context: Context that should be split.
        :return: Iterable of tuples
            tokenized prefix - tokenized prefix of input string that is same for all samples in a document (title)
            tokenized - tokenized input string that should be put on the input of a model
            keyphrase spans as INCLUSIVE token offsets
        :raise RuntimeError: When the sample couldn't be split.
        """

        parts = context.split("\t")

        context_content, spans = parts[0], []

        if len(parts) > 1:
            spans = parts[1:]

        tokens_context_offset_mapping = self._tokenizer.encode_plus(context_content,
                                                                    return_offsets_mapping=True,
                                                                    return_attention_mask=False,
                                                                    return_token_type_ids=False,
                                                                    add_special_tokens=False)['offset_mapping']

        free_space = self._tokenizer.model_max_length

        if len(title) > 0:
            free_space -= len(self._tokenizer.tokenize(title))
            free_space -= self._tokenizer.num_special_tokens_to_add(True)
        else:
            free_space -= self._tokenizer.num_special_tokens_to_add(False)

        spans = CzechLibrary.transform_spans(spans)

        # make splits
        if len(tokens_context_offset_mapping) > free_space > 0:
            parts = self.fair_split(free_space, tokens_context_offset_mapping, spans)
        else:
            # no free space left for a context or no splitting is required in both cases we provide the whole thing
            # and do no splitting at all.

            parts = [(0, len(context_content))]

        # assemble inputs
        res = []
        for part_char_start, part_char_end in parts:
            parts_spans = [
                (s - part_char_start, e - part_char_start) for s, e in spans if
                part_char_start <= s <= e <= part_char_end
            ]

            res.append(self._assemble_input(title, context_content[part_char_start:part_char_end], parts_spans))

        return res


class CzechLibrary(Dataset):
    """
    Dataset of keyphrases created from data from czech libraries.

    Format of sample:
        Context followed with voluntary number of spans defined by character offset of start and end:
            context (\t char_offset_of_span_start char_offset_of_span_end)*
    Example of sample:
        Neznám toho moc z kapitoly občanské právo.\t27 41

    Beware that when a dataset sample is longer and wouldn't fit to the model input it will be split and each split
    will act as a separate sample (model sample). The length of dataset return number of model samples and also
    __getitem__ indexes and returns model samples.

    :ivar path_to: Path to dataset file
    :vartype path_to: str
    :ivar tokenizer: Tokenizer that is used for context tokenization.
    :vartype tokenizer: PreTrainedTokenizer
    :ivar gt_span_universe: False deactivates creation of span universe matrix for each sample.
            Span universe boolean matrix for an input sequence of size N is a matrix U of size NxN.
            Where each span is identified by True on its start and end token indices.
    :vartype gt_span_universe: bool
    """

    def __init__(self, path_to: str, tokenizer: PreTrainedTokenizerFast, verbose: bool = True,
                 gt_span_universe: bool = True, add_title: bool = True):
        """
        Initialization of dataset.

        :param path_to: Path to file with dataset.
        :param tokenizer: Tokenizer that is used for context tokenization.
        :param verbose: False hides progress bar when the index is created.
        :param gt_span_universe: False deactivates creation of span universe matrix for each sample.
            Span universe boolean matrix for an input sequence of size N is a matrix U of size NxN.
            Where each span is identified by True on its start and end token indices.
        :param add_title: True adds the title to the input (before context [separator is used]).
        :raise RuntimeError: Loaded index was created for different dataset or different preprocessed dataset.
            Beware that this is just naive check on dataset size in bytes, so this will not catch all differences.
        """

        self.path_to = path_to
        self.tokenizer = tokenizer
        self._file_descriptor = None
        self._opened_in_process_with_id = None
        self.gt_span_universe = gt_span_universe
        self.add_title = add_title

        self.preprocessed = self.preprocess(self.path_to, self.tokenizer, self.add_title, verbose)

        if self.preprocessed is None:  # the index already exists
            # raises RuntimeError when there is a mismatch with prep. dataset
            self.preprocessed = PreprocessedDataset(self.path_to_preprocessed)
            # raises RuntimeError when there is a mismatch with orig. dataset
            self.preprocessed.check_match_with(self.path_to)

    @classmethod
    def preprocess(cls, path_to: str, tokenizer: PreTrainedTokenizerFast, title: bool, verbose: bool = True,
                   workers: int = 0) -> Optional[PreprocessedDataset]:
        """
        Creates preprocessed dataset if it does not exists.

        Warning: It does not guarantees that an already existing one was created for current dataset.

        :param path_to: Path to original dataset.
        :param tokenizer: The tokenizer that will be used for index creation.
        :param title: Whether the title will be used.
        :param verbose: False hides progress bar when the index is created.
        :param workers: Number of additional parallel workers.
        :return: Created preprocessed dataset or None when one already exists.
        """
        path_to_prep = cls.path_to_preprocessed_dataset(path_to, tokenizer, title)
        if not os.path.isfile(path_to_prep):
            # suppress warnings: Token indices sequence length is longer than the specified maximum sequence
            transformers_logger_level = logging.getLogger("transformers.tokenization_utils_base").level

            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

            prep = PreprocessedDataset.from_dataset(
                path_to=path_to,
                res_to=path_to_prep,
                use_title=title,
                splitter=CzechLibrarySampleSplitter(tokenizer),
                verbose=verbose,
                workers=workers
            )

            logging.getLogger("transformers.tokenization_utils_base").setLevel(transformers_logger_level)

            return prep

        return None

    @staticmethod
    def path_to_preprocessed_dataset(path_to: str, tokenizer: PreTrainedTokenizerFast, title: bool):
        """
        Path to file where the preprocessed dataset should be saved.

        :param path_to: Path to original dataset.
        :param tokenizer: The tokenizer that will be used for index creation.
        :param title: Whether the title will be used.
        :return: path to the index
        """
        path = Path(path_to)

        return os.path.join(str(path.parents[0]),
                            path.stem + "_"
                            + Path(tokenizer.name_or_path).stem + "_"
                            + ("title" if title else "no_title")
                            + ".prep")

    @property
    @functools.lru_cache()
    def path_to_preprocessed(self):
        """
        Path to file where the preprocessed dataset should be saved.
        Same as `.path_to_index_for_dataset` but uses current parameters of an instance.
        """

        return self.path_to_preprocessed_dataset(self.path_to, self.tokenizer, self.add_title)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> "CzechLibrary":
        """
        Opens the dataset if it is closed, else it is just empty operation.

        :return: Returns the object itself.
        :rtype: CzechLibrary
        """

        if self._file_descriptor is None:
            self._file_descriptor = open(self.path_to, 'rb')
            self._opened_in_process_with_id = os.getpid()

        self.preprocessed.open()
        return self

    def close(self):
        """
        Closes the dataset.
        """

        if self._file_descriptor is not None:
            self._file_descriptor.close()
            self._file_descriptor = None
            self._opened_in_process_with_id = None

        self.preprocessed.close()

    def _reopen_if_needed(self):
        """
        Reopens itself if the multiprocessing is activated and this dataset was opened in parent process.
        """

        if os.getpid() != self._opened_in_process_with_id or self._file_descriptor is None:
            self.close()
            self.open()

    def __len__(self) -> int:
        """
        number of model samples
        """
        return len(self.preprocessed)

    def line(self, offset: int) -> str:
        """
        Get line from dataset file that begins on given offset.

        :param offset: Offset of a line for file descriptor.
        :return: the line
        """
        self._reopen_if_needed()

        self._file_descriptor.seek(offset)
        return self._file_descriptor.readline().decode("utf-8").strip()

    def part(self, offset, length) -> str:
        """
        Reads part of dataset file.

        :param offset: File offset to the start.
        :param length: How many bytes should be read from given str offset.
        :return: read string
        """
        self._reopen_if_needed()
        self._file_descriptor.seek(offset)
        # json loads is here because of the tabulators
        return json.loads('"' + self._file_descriptor.read(length).decode("utf-8") + '"', strict=False).strip()

    @staticmethod
    def transform_spans(spans: List[str]) -> List[Tuple[int, int]]:
        """
        Transformation of spans in string form
            "0 10"
        to tuple of two integers:
            (0, 10)
        :param spans: spans in string form
        :return: spans as integer tuples
        """
        res = []
        for s in spans:
            parts = s.split()
            res.append((int(parts[0]), int(parts[1])))

        return res

    @staticmethod
    def char_2_token_lvl_span(s: Tuple[int, int], char_offset_mapping: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Converts span from character level representation to token level representation.

        :param s: Span in character level representation.
        :param char_offset_mapping: Character level span for each token in an input.
        :return: Span in token level representation.
            WARNING input span is right opened interval, but the output in token level is closed interval.
        :raise RuntimeError: When mapping is not possible.
        """
        token_start = None
        token_end = None

        for i_tok, (st, en) in enumerate(char_offset_mapping):
            if token_start is None and en > s[0]:
                token_start = i_tok
            if token_end is None and en >= s[1]:
                token_end = i_tok
            if token_start is not None and token_end is not None:
                return token_start, token_end

        # we are matching always to the nearest token
        # but if the character offset is out of sequence than this might happen
        raise RuntimeError(f"Couldn't map span in character representation {s} to token level representation for "
                           f"tokens sequence {char_offset_mapping}")

    def __getitem__(self, i) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get i-th model sample.

        :param i:  identifier of model sample
        :return: tuple (line_offset, tokens, spans)
        """

        line_offset, tokens, spans = self.preprocessed[i]

        return line_offset, tokens, spans

    def collate_fn(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Batch:
        """
        Creates batch from list of samples.

        :param samples: Samples in form of list of tuples
            (line_offset, tokens, spans)
        :return: batch created from inputs
        """
        max_len = min(max(s[1].shape[0] for s in samples), self.tokenizer.model_max_length)  # no longer than max size

        batch_line_offset = []
        batch_tokens = torch.full((len(samples), max_len), self.tokenizer.pad_token_id)
        batch_attention_mask = torch.zeros_like(batch_tokens)

        batch_spans = []

        for i, (line_offset, tokens, spans) in enumerate(samples):
            # truncation
            tokens = \
                tokens[:self.tokenizer.model_max_length] if len(tokens) > self.tokenizer.model_max_length else tokens

            batch_tokens[i, :tokens.shape[0]] = tokens
            batch_attention_mask[i, :tokens.shape[0]] = 1

            batch_line_offset.append(line_offset)
            batch_spans.append(spans)

        if self.gt_span_universe:
            batch_gt_span_universe = torch.zeros(len(samples), max_len, max_len, dtype=torch.bool)

            for i, spans in enumerate(batch_spans):
                for token_start, token_end in spans:
                    try:
                        batch_gt_span_universe[i, token_start, token_end] = True
                    except IndexError:
                        # this span was truncated out
                        pass

        else:
            batch_gt_span_universe = None

        return Batch(tokens=batch_tokens,
                     attention_masks=batch_attention_mask,
                     gt_span_universes=batch_gt_span_universe,
                     line_offsets=batch_line_offset,
                     tokenizer=self.tokenizer)


class CzechLibraryDataModule(LightningDataModule):
    """
    Data module for CzechLibrary dataset.

    :ivar config: used configuration
    :vartype config: Dict[str, Any]
    :ivar train: train dataset
    :vartype train: Optional[CzechLibrary]
    :ivar val: validation dataset
    :vartype val: Optional[CzechLibrary]
    :ivar test: test dataset
    :vartype test: Optional[CzechLibrary]
    :ivar verbose: True activates verbose log.
    :vartype verbose: bool
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialization of data module.

        :param config: dictionary with configuration
            Expects:
                [training|validation|testing][activate]
                    True activates creation of training|validation|testing dataset.
                [training|validation|testing][dataset][path]
                    path to training|validation dataset
                [training|validation|testing][dataset][workers]
                    values > 0 activates multi process reading of dataset and the value determines
                    number of subprocesses that will be used for reading (the main process is not counted).
                    If == 0 than the single process processing is activated.
                [training|validation|testing][batch]
                    batch size for training|validation|testing

                [transformers][tokenizer]
                    fast tokenizer that should be used
                    (see https://huggingface.co/transformers/main_classes/tokenizer.html)
                [transformers][cache]
                    Cache where the transformers library will save the models.
        """
        super().__init__()
        self.config = config
        self.train = None
        self.val = None
        self.test = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["transformers"]["tokenizer"], use_fast=True,
                                                           cache_dir=self.config["transformers"]["cache"],
                                                           local_files_only=self.config["transformers"][
                                                               "local_files_only"])
        except ValueError:
            # try again, might be connection error
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["transformers"]["tokenizer"], use_fast=True,
                                                           cache_dir=self.config["transformers"]["cache"],
                                                           local_files_only=True)

        self.verbose = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.config["training"]["batch"],
            shuffle=True,
            num_workers=self.config["training"]["dataset"]["workers"],
            pin_memory=True,
            collate_fn=self.train.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.config["validation"]["batch"],
            shuffle=False,
            num_workers=self.config["validation"]["dataset"]["workers"],
            pin_memory=True,
            collate_fn=self.val.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.config["testing"]["batch"],
            shuffle=False,
            num_workers=self.config["testing"]["dataset"]["workers"],
            pin_memory=True,
            collate_fn=self.test.collate_fn
        )

    def prepare_data(self):
        # let's make preprocessing or load and check already preprocessed ones
        for dataset in ["training", "validation", "testing"]:
            if dataset in self.config:
                if "activate" not in self.config[dataset] or self.config[dataset]["activate"]:
                    prep = CzechLibrary.preprocess(path_to=self.config[dataset]["dataset"]["path"],
                                                   tokenizer=self.tokenizer,
                                                   title=self.config[dataset]["dataset"]["add_title"],
                                                   verbose=self.verbose,
                                                   workers=self.config[dataset]["dataset"]["prep_workers"])
                    if prep is None:
                        # naive check that existing dataset is the same
                        prep_path = CzechLibrary.path_to_preprocessed_dataset(
                            path_to=self.config[dataset]["dataset"]["path"],
                            tokenizer=self.tokenizer,
                            title=self.config["training"]["dataset"][
                                "add_title"])

                        # raises RuntimeError when there is a mismatch with prep. dataset
                        prep = PreprocessedDataset(prep_path)
                        # raises RuntimeError when there is a mismatch with orig. dataset
                        prep.check_match_with(self.config[dataset]["dataset"]["path"])

    def setup(self, stage=None):
        if "training" in self.config:
            self.train = CzechLibrary(path_to=self.config["training"]["dataset"]["path"],
                                      tokenizer=self.tokenizer, verbose=self.verbose,
                                      add_title=self.config["training"]["dataset"]["add_title"]).open()

        if "validation" in self.config:
            self.val = CzechLibrary(path_to=self.config["validation"]["dataset"]["path"],
                                    tokenizer=self.tokenizer, verbose=self.verbose,
                                    add_title=self.config["validation"]["dataset"]["add_title"]).open()

        if "testing" in self.config and (
                "activate" not in self.config["testing"] or self.config["testing"]["activate"]):
            self.test = CzechLibrary(path_to=self.config["testing"]["dataset"]["path"], tokenizer=self.tokenizer,
                                     gt_span_universe=False, verbose=self.verbose,
                                     add_title=self.config["testing"]["dataset"]["add_title"]).open()

    def teardown(self, stage: Optional[str] = None):
        for dataset in [self.train, self.val, self.test]:
            if dataset is not None:
                dataset.close()

    def transfer_batch_to_device(self, batch: Batch, device: torch.device, dataloader_idx: int) -> Batch:
        return batch.to(device)
