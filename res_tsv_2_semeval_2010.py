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

from porter_stemmer import PorterStemmer


def main():
    pred_csv = pd.read_csv(sys.argv[1], sep="\t")
    p = PorterStemmer()

    key_field = sys.argv[3] if len(sys.argv) > 3 else "predicted"

    with open(sys.argv[2], "w") as f:
        for _, row in pred_csv.iterrows():
            if pd.isna(row[key_field]):
                predictions = ""
            else:
                keyphrases = row[key_field].split('█')
                stemmed_keyphrases = []
                for k in keyphrases:
                    output = ""
                    word = ""
                    for c in k:
                        if c.isalpha():
                            word += c.lower()
                        else:
                            if word:
                                output += p.stem(word, 0, len(word) - 1)
                                word = ""
                            output += c.lower()
                    if word:
                        output += p.stem(word, 0, len(word) - 1)

                    if output not in stemmed_keyphrases:
                        stemmed_keyphrases.append(output)

                predictions = ','.join(stemmed_keyphrases[:15])
            print(f"{str(row['uuid'])} : {predictions}", file=f)


if __name__ == '__main__':
    main()
