# -*- coding: UTF-8 -*-
""""
Created on 30.09.21
Preprocessing of datasets.

:author:     Martin Dočekal
"""
import io
import json
import multiprocessing
import os
import re
from abc import ABC, abstractmethod
from functools import partial
from typing import Generator, Tuple, List, Optional, Iterable, Any, Dict, Callable, TypeVar

import numpy as np
import torch
from tqdm import tqdm


class SampleSplitter(ABC):
    """
    Abstract class for splitting dataset samples into model samples.

    """

    @abstractmethod
    def split(self, title: str, context: str) -> Iterable[Tuple[Optional[np.array], np.array, np.array]]:
        """
        Splits dataset sample into parts (model samples) which fit into a model.

        NOTE TO THE used numpy arrays:
            As for now there is a problem (https://github.com/pytorch/pytorch/issues/65198) with sending tensors via
            multiprocessing queue the numpy arrays are used.
            TODO: Transform that in the future

        :param title: Title of document
        :param context: Context that should be split.
        :return: Iterable of tuples
            tokenized prefix - tokenized prefix of input string that is same for all samples in a document (title)
            tokenized - tokenized input string that should be put on the input of a model
            keyphrase spans as INCLUSIVE token offsets
        :raise RuntimeError: When the sample couldn't be split.
        """
        pass


T = TypeVar('T')


class SampleConvertor:
    byteorder = 'big'
    """byte order that will be used for saving"""

    line_offset_bytes = 8
    """number of bytes representing a line offset"""

    token_bytes = 3
    """number of bytes representing a token"""

    keyphrase_span_offset_bytes = 2
    """number of bytes representing a keyphrase span offset"""

    counter_bytes = 2
    """Number of bytes for a counter that states number of following elements of given kind."""

    @classmethod
    def sample_2_bytes(cls, line_offset: int, tokenized: np.array, keyphrase_spans: np.array) -> bytes:
        """
        Conversion of sample to its byte representation.

        :param line_offset: offset of document to which this sample belongs
        :param tokenized: tokenized input that can go straight to the model
        :param keyphrase_spans: keyphrase span offset on token lvl
        :return: byte representation
        """

        res = line_offset.to_bytes(cls.line_offset_bytes, byteorder=cls.byteorder)

        res += len(tokenized).to_bytes(cls.counter_bytes, byteorder=cls.byteorder)
        for token in tokenized:
            res += int(token).to_bytes(cls.token_bytes, byteorder=cls.byteorder)

        res += len(keyphrase_spans).to_bytes(cls.counter_bytes, byteorder=cls.byteorder)
        for start, end in keyphrase_spans:
            res += int(start).to_bytes(cls.keyphrase_span_offset_bytes, byteorder=cls.byteorder)
            res += int(end).to_bytes(cls.keyphrase_span_offset_bytes, byteorder=cls.byteorder)

        return res

    @classmethod
    def bytes_2_sample(cls, b_repr: bytes) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Conversion from byte representation to the sample.

        :param b_repr: byte representaiton o sample
        :return: tuple:
            line_offset: offset of document to which this sample belongs
            tokenized: tokenized input that can go straight to the model
            keyphrase_spans: keyphrase span offset on token lvl
        """
        line_offset = int.from_bytes(b_repr[:cls.line_offset_bytes], byteorder=cls.byteorder)

        read_offset = cls.line_offset_bytes

        res = []
        for byte_size, get_elem in [(cls.token_bytes, cls.convert_token),
                                    (2*cls.keyphrase_span_offset_bytes, cls.convert_keyphrase_span)]:
            cnt = int.from_bytes(b_repr[read_offset:read_offset+cls.counter_bytes], byteorder=cls.byteorder)
            read_offset += cls.counter_bytes
            end = read_offset + cnt * byte_size
            res.append(cls.read_n_elements(get_elem, byte_size, b_repr[read_offset:end]))
            read_offset = end

        return line_offset, torch.tensor(res[0]), torch.tensor(res[1])

    @classmethod
    def convert_token(cls, bs: bytes) -> int:
        """
        Convert token from its byte representation.

        :param bs: byte sequence
        :return: token
        """
        return int.from_bytes(bs, byteorder=cls.byteorder)

    @classmethod
    def convert_keyphrase_span(cls, bs: bytes) -> Tuple[int, int]:
        """
        Convert keyphrase span from its byte representation.

        :param bs: byte sequence
        :return: keyphrase span
        """
        return (
            int.from_bytes(bs[:cls.keyphrase_span_offset_bytes], byteorder=cls.byteorder),
            int.from_bytes(bs[cls.keyphrase_span_offset_bytes:], byteorder=cls.byteorder)
        )

    @staticmethod
    def read_n_elements(get_elem: Callable[[bytes], T], size: int, seq: bytes) -> List[T]:
        """
        Read n integers of given byte size from byte sequence.

        :param get_elem: Converts bytes sequence to a target element.
        :param size: size of single element
        :param seq: sequence of elements
        :return: List of parsed elements.
        """

        return [get_elem(seq[s_offset:s_offset+size]) for s_offset in range(0, len(seq), size)]


class PreprocessedDataset:
    """
    Class for reading preprocessed dataset and its creation.

    USAGE:
        with PreprocessedDataset(...) as d:
            print(d[0])

        d = PreprocessedDataset(...).open()
        print(d[0])
        d.close()
    """

    INDEX_EXT = ".index"
    """Extension to the preprocessed dataset file for index file."""

    def __init__(self, path_to: str, index: Optional[Dict[str, Any]] = None):
        """
        Initialization of preprocessed dataset.

        :param path_to: path to preprocessed dataset
        :param index: optionally you can provided an index else it will be loaded from file
        :raise RuntimeError: When corresponding index was created for different preprocessed dataset.
        :raise FileNotFoundError: When the index file doesn't exists.
        """

        self._path_to = path_to

        self._index = torch.load(self.path_to + self.INDEX_EXT) if index is None else index

        if os.path.getsize(self._path_to) != self._index["preprocessed_dataset_size"]:
            raise RuntimeError("Loaded index was created for different preprocessed dataset. "
                               "Beware that this is just naive check on dataset size in bytes, so this will not catch "
                               "all differences.")

        self._dataset_file = None
        self._opened = False
        self.opened_in_process_with_id = None

    @property
    def path_to(self) -> str:
        """
        Path to preprocessed dataset file.
        """
        return self._path_to

    @property
    def opened(self) -> bool:
        """
        True when prep. dataset is opened.
        """
        return self._opened

    def __len__(self):
        return len(self._index["samples_index"])

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _get_sample_len(self, idx: int) -> int:
        """
        Get preprocessed model sample length.

        :param idx: index of model sample
        :return: byte len
        """
        b_offset = self._index["samples_index"][idx]
        if idx == len(self._index["samples_index"]) - 1:
            end_off = self._index["preprocessed_dataset_size"]
            return end_off - b_offset
        else:
            return self._index["samples_index"][idx + 1] - b_offset

    def doc_offset(self, line_offset: int) -> int:
        """
        Get document offset from line offset.

        :param line_offset: number of bytes to the beginning of line
        :return: document offset
        """
        return self._index["line_offsets"].index(line_offset)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Model sample on given index.

        :param idx: the offset of given sample
        :return: line_offset, tokenized, keyphrase spans (with INCLUSIVE offsets)
        :raise RuntimeError: When you forgot to open this dataset.
        """

        if not self.opened:
            RuntimeError("Please open this dataset before you use it.")

        self._reopen_if_needed()  # for the multiprocessing case

        # get the sample
        b_offset = self._index["samples_index"][idx]
        b_len = self._get_sample_len(idx)
        return self._read_entity_from_dataset(b_offset, b_len)

    def _read_entity_from_dataset(self, o: int, length: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Reads an entity saved on given offset and having the given length.

        :param o: offset when the reading should start
        :param length: length in bytes of the reading
        :return: the loaded model sample
        """

        self._dataset_file.seek(o)
        data = self._dataset_file.read(length)
        sample = SampleConvertor.bytes_2_sample(data)
        if self._index["doc_samples_prefix"] is not None:
            sample = (
                sample[0],
                torch.hstack([self._index["doc_samples_prefix"][self.doc_offset(sample[0])], sample[1]]),
                sample[2]
            )
        return sample

    def _reopen_if_needed(self):
        """
        Reopens itself if the multiprocessing is activated and this dataset was opened in parent process.
        """

        if os.getpid() != self.opened_in_process_with_id and self._opened:
            self.close()
            self.open()

    def open(self) -> "PreprocessedDataset":
        """
        Open the prep. dataset if it was closed, else it is just empty operation.

        :return: Returns the object itself.
        """

        if self._dataset_file is None:
            self._dataset_file = open(self.path_to, "rb")

        self.opened_in_process_with_id = os.getpid()
        self._opened = True
        return self

    def close(self):
        """
        Closes the dataset.
        """

        if self._dataset_file is not None:
            self._dataset_file.close()
            self._dataset_file = None

        self.opened_in_process_with_id = None
        self._opened = False

    @classmethod
    def _read_dataset(cls, path_to: str, verbose: bool = True) -> Generator[Tuple[str, int], None, None]:
        """
        Reads given dataset.

        :param path_to: path to dataset you want to read
        :param verbose: False hides progress bar.
        :return: Generates
            line string
            file offset of the line
        """

        num_of_bytes = os.path.getsize(path_to)
        with tqdm(total=num_of_bytes, desc=f"Reading dataset for preprocessing {path_to}",
                  unit="byte", disable=not verbose) as pBar:
            with open(path_to, "rb") as f:
                line = f.readline()
                line_offset = 0
                while line:
                    yield line, line_offset
                    line_offset = f.tell()
                    pBar.update(f.tell() - pBar.n)
                    line = f.readline()

    @classmethod
    def _proc_doc_line(cls, line: str, line_offset: int, use_title: bool, splitter: SampleSplitter) \
            -> Tuple[int, Optional[np.array], List[Tuple[np.array, np.array]]]:
        """
        Processes single line representing document.

        :param line: line in json format representing document
        :param line_offset: file offset of the line
        :param use_title: whether the title should also be used
        :param splitter: splitter for splitting dataset lines to model samples.
        :return: Model samples for given document
            Tuples in format:
                line_offset, prefix (tokenized title), List[tokenized, keyphrase spans]
        """
        res = re.search(b'"query": "(.+?)", "keyphrases": \[', line)

        if use_title:
            title = res.group(1).decode("utf-8")
        else:
            title = ""

        model_samples = []
        prefix_tokens = None
        contexts_match = re.search(b'"contexts": \[(.+)\]}$', line)
        if contexts_match:
            # non empty contexts
            contexts = contexts_match.group(1)

            for c_match in re.finditer(b'(?<!\\\\)(?:\\\\\\\\)*"(.+?)(?<!\\\\)(?:\\\\\\\\)*"', contexts):
                c = c_match.group(1).decode("utf-8")

                # translate escaped tabulator
                c = json.loads('"' + c + '"', strict=False).strip()

                for pref_tok, tokens, spans in splitter.split(title, c.strip()):
                    if prefix_tokens is None and pref_tok is not None:
                        prefix_tokens = pref_tok
                    model_samples.append((tokens, spans))

        return line_offset, prefix_tokens, model_samples

    @classmethod
    def _proc_doc_line_wrapper(cls, line_data: Tuple[str, int], use_title: bool, splitter: SampleSplitter) \
            -> Tuple[int, Optional[np.array], List[Tuple[np.array, np.array]]]:
        """
        This is just wrapper for :py:meth:DatasetIndex._proc_doc_line that is ment to be used to pass results from
        :py:meth:DatasetIndex._read_dataset to :py:meth:DatasetIndex._proc_doc_line .

        :param line_data: Results from :py:meth:DatasetIndex._read_dataset in form of tuple:
            line content, file offset of the line
        :param use_title: whether the title should also be used
        :param splitter: splitter for splitting dataset lines to model samples.
        :return: parse model samples
            Tuples in format:
                line_offset, prefix (tokenized title), List[tokenized, keyphrase spans]
        """
        return cls._proc_doc_line(
            line=line_data[0],
            line_offset=line_data[1],
            use_title=use_title,
            splitter=splitter
        )

    @classmethod
    def from_dataset(cls, path_to: str, res_to: str, use_title: bool, splitter: SampleSplitter, verbose: bool = True,
                     workers: int = 0) -> "PreprocessedDataset":
        """
        Creates index from dataset file.

        :param path_to: Path to dataset file in jsonl format.
        :param res_to: Path where preprocessed results should be saved.
            It automatically creates another file with the same path as this but with INDEX_EXT extension which contains
            the index.
        :param use_title: whether the title should also be used
        :param splitter: Splitter for splitting dataset lines to model samples.
        :param verbose: False hides progress bar.
        :param workers: Number of additional parallel workers.
        :return: newly created dataset
        """
        index = {
            "original_dataset_size": os.path.getsize(path_to),
            "line_offsets": [],
            "doc_samples_prefix": [] if use_title else None,   # tokenized titles
            "samples_index": []
        }

        pool = None

        if workers == 0:
            proc_map = map
        else:
            pool = multiprocessing.Pool(processes=workers)
            proc_map = pool.imap

        with open(res_to, "wb") as f:
            for line_offset, samples_prefix, document_samples in proc_map(
                    partial(cls._proc_doc_line_wrapper, use_title=use_title, splitter=splitter),
                    cls._read_dataset(path_to, verbose)):

                index["line_offsets"].append(line_offset)
                if index["doc_samples_prefix"] is not None:
                    index["doc_samples_prefix"].append(
                        torch.tensor([]) if samples_prefix is None else torch.from_numpy(samples_prefix)
                    )

                if len(document_samples) > 0:
                    for sample in document_samples:
                        sample = (  # tokenized, keyphrase spans
                            line_offset, sample[0], sample[1]
                        )

                        index["samples_index"].append(f.tell())
                        f.write(SampleConvertor.sample_2_bytes(sample[0], sample[1], sample[2]))
                else:
                    # placeholder sample for a document
                    f.write(SampleConvertor.sample_2_bytes(line_offset, np.array([]), np.array([])))

        if pool is not None:
            pool.close()
            pool.join()

        index["preprocessed_dataset_size"] = os.path.getsize(res_to)
        index["samples_index"] = torch.tensor(index["samples_index"])

        torch.save(index, res_to + cls.INDEX_EXT)

        return cls(res_to, index)

    def check_match_with(self, path_to: str):
        """
        Naive check of dataset size whether this index matches to the dataset on given path.

        :param path_to: path to original dataset
        :raise RuntimeError: When index and dataset do not match.
        """

        if os.path.getsize(path_to) != self._index["original_dataset_size"]:
            raise RuntimeError("Loaded index was created for different dataset. Beware that this is just naive "
                               "check on dataset size in bytes, so this will not catch all differences.")
