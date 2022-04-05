# -*- coding: UTF-8 -*-
""""
Created on 30.09.21

:author:     Martin Dočekal
"""
from transformers import AutoTokenizer

from qbek.datasets.czech_library import CzechLibrarySampleSplitter
from qbek.datasets.preprocessing import PreprocessedDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

splitter = CzechLibrarySampleSplitter(tokenizer)
prepr = PreprocessedDataset.from_dataset("dataset.txt", "dataset.prep", False, splitter)
