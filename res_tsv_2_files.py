#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 21.10.21
Converts .tsv file with predictions to separate file named as uuid containing predictions.

Arguments:
    1   path .tsv file with predictions
        Should contain uuid and predicted columns
            predictions must be separated by █
    2   path to directory where files will be saved

:author:     Martin Dočekal
"""
import sys
from pathlib import Path

import pandas as pd


def main():
    pred_csv = pd.read_csv(sys.argv[1], sep="\t")

    dir_path = Path(sys.argv[2])
    dir_path.mkdir(parents=True, exist_ok=True)
    for _, row in pred_csv.iterrows():
        with open(dir_path.joinpath(str(row["uuid"])+".res"), "w") as f:
            if not pd.isna(row["predicted"]):
                for keyphrase in row["predicted"].split("█"):
                    print(keyphrase, file=f)


if __name__ == '__main__':
    main()
