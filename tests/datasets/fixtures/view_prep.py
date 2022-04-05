# -*- coding: UTF-8 -*-
""""
Created on 01.10.21
Viewer of preprocessed dataset.

:author:     Martin Dočekal
"""
import pickle
import sys

import torch
from transformers import AutoTokenizer

from qbek.datasets.preprocessing import PreprocessedDataset

tok_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(tok_name)
print(f"Using tokenizer {tok_name}. Change it in the code if needed.")

if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
else:
    print("Please enter path to preprocessed dataset file:")
    for line in sys.stdin:
        dataset_path = line[:-1]
        break

dataset = PreprocessedDataset(dataset_path)

with open(dataset_path, "rb") as prep_f, dataset as prep_dataset:
    print(len(dataset._index["samples_index"]))

    for i, start_offset in enumerate(dataset._index["samples_index"]):
        prep_f.seek(start_offset)
        length = dataset._get_sample_len(i)
        print(prep_f.read(length))
        prep_sample = prep_dataset[i]
        print(prep_sample)
        print(tokenizer.decode(prep_sample[1]))
        print("\t".join(tokenizer.decode(prep_sample[1][key_span[0]:key_span[1]+1]) for key_span in prep_sample[2]))
