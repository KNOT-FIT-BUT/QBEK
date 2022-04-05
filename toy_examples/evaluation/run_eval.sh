#!/bin/bash

# creates predictions and their evaluation
# the evaluation is not done in the SemEval fashion, if you are interested in those metrics you should
# investigate the metrics package
../../run.py eval eval_config.py

# let's convert the prediction into format that accepts the SemEval script
# there is also czech variant of conversion script res_tsv_2_semeval_2010_czech.py, that uses czech lemmatizer
../../res_tsv_2_semeval_2010.py eval_predictions.tsv semeval_predictions.txt

# run SemEval evaluation
../data/res/performance.pl semeval_predictions.txt ../data/res/ > eval_semeval.txt

