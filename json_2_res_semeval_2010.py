#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 21.10.21
Converts dataset .json file to format that is suitable for SemEval 2010 evaluator

Arguments:
    1   path to dataset .json
    2   path of file where the results should be saved.
    3   (optional) field with keyphrases that should be converted, default is predicted

:author:     Martin Dočekal
"""
import json
import sys

from qbek.lemmatizer import MorphoditaLemmatizer, PorterStemmer


def main():
    p = PorterStemmer()

    key_field = sys.argv[3] if len(sys.argv) > 3 else "keyphrases"

    with open(sys.argv[1], "r") as f, open(sys.argv[2], "w") as resF:
        for line in f:
            row = json.loads(line)
            stem_keyphrases = set()
            for k in row[key_field]:
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

                stem_keyphrases.add(output)

            keyphrases = ','.join(sorted(stem_keyphrases))

            print(f"{str(row['uuid'])} : {keyphrases}", file=resF)


if __name__ == '__main__':
    main()
