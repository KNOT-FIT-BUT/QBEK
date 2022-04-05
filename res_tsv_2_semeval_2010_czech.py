#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 21.10.21
Converts .tsv file with predictions to format that is suitable for SemEval 2010 evaluator

Arguments:
    1   path .tsv file with predictions
        Should contain uuid and predicted columns
            predictions must be separated by █
    2   path of file where the results should be saved.
    3   (optional) field with keyphrases that should be converted, default is predicted

:author:     Martin Dočekal
"""
import sys

import pandas as pd

from qbek.lemmatizer import MorphoditaLemmatizer


def main():
    pred_csv = pd.read_csv(sys.argv[1], sep="\t")
    p = MorphoditaLemmatizer()

    key_field = sys.argv[3] if len(sys.argv) > 3 else "predicted"

    with open(sys.argv[2], "w") as f:
        for _, row in pred_csv.iterrows():
            if pd.isna(row[key_field]):
                predictions = ""
            else:
                keyphrases = row[key_field].split('█')
                lemma_keyphrases = []
                for k in keyphrases:
                    l_k = " ".join(p.lemmatize(k.split())).lower()
                    if l_k not in lemma_keyphrases:
                        lemma_keyphrases.append(l_k)

                predictions = ','.join(lemma_keyphrases[:15])
            print(f"{str(row['uuid'])} : {predictions}", file=f)


if __name__ == '__main__':
    main()
